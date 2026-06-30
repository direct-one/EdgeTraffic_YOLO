from pathlib import Path
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


# 실행 위치와 관계없이 현재 파일과 같은 폴더의 TensorRT 엔진을 사용합니다.
ENGINE_PATH = Path(__file__).resolve().parent / "vehicle5_yolo11n_e250.engine"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MODEL_IMAGE_SIZE = 640

# ── 검출/신뢰도 홀드 필터 설정 ─────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3       # 0.3 이상인 경우 검출
LOW_CONFIDENCE_THRESHOLD = 0.1   # 추적 단계에서 일단 이 값까지 검출을 받아옴
IOU_THRESHOLD = 0.45             # NMS IoU
HOLD_SECONDS = 2                 # 검출이 사라져도 이 시간 동안은 유지
MATCH_IOU_THRESHOLD = 0.3        # 직전 박스와 매칭할 때 사용하는 IoU
CROSS_CLASS_IOU_THRESHOLD = 0.5  # 다른 클래스의 중복으로 보는 IoU
# ──────────────────────────────────────────────────────────────

# ── 정차(emergency) / 주정차 금지(illegal) 판정 설정 ───────────
STATIONARY_LIMIT_SECONDS = 5.0    # 이 시간 이상 멈춰 있으면 emergency 로 판정
ILLEGAL_LIMIT_SECONDS = 10.0      # 금지구역 안에 이 시간 이상 머물면 illegal 로 판정
MOVEMENT_THRESHOLD_PX = 20.0      # 중심 좌표 이동이 이 거리(px) 이내면 정지로 간주
TRACK_TIMEOUT_SECONDS = 5.0       # 이 시간 동안 추적이 끊기면 상태 삭제(메모리 정리)
CONGESTION_VEHICLE_COUNT = 8      # 화면에 잡힌 전체 차량이 이 수 이상이면 정체 구간으로 보고 emergency 표시 안 함
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"  # 자동 캡쳐 저장 폴더
# ──────────────────────────────────────────────────────────────

VEHICLE_NAMES = {
    0: "truck",
    1: "trailer",
    2: "bus",
    3: "car",
    4: "motorcycle",
}
WINDOW_NAME = "Vehicle Detection"


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_in_box(center, box):
    cx, cy = center
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def is_cross_class_duplicate(box_a, class_a, box_b, class_b):
    if class_a == class_b:
        return False

    center_a = box_center(box_a)
    center_b = box_center(box_b)
    centers_overlap = center_in_box(center_a, box_b) and center_in_box(center_b, box_a)
    boxes_overlap = box_iou(box_a, box_b) >= CROSS_CLASS_IOU_THRESHOLD

    return centers_overlap or boxes_overlap


def remove_cross_class_duplicate_boxes(keep_indexes, boxes_xyxy, class_ids, confidences):
    """
    같은 위치에 서로 다른 클래스로 중복 검출된 박스 중,
    신뢰도가 낮은 쪽을 제거
    """
    filtered_indexes = []

    for index in sorted(keep_indexes, key=lambda i: confidences[i], reverse=True):
        box = boxes_xyxy[index]
        class_id = class_ids[index]

        is_duplicate = any(
            is_cross_class_duplicate(
                box,
                class_id,
                boxes_xyxy[kept_index],
                class_ids[kept_index],
            )
            for kept_index in filtered_indexes
        )
        if not is_duplicate:
            filtered_indexes.append(index)

    return filtered_indexes


def filter_boxes_with_confidence_hold(result, recent_boxes, current_time):
    """
    낮은 conf 로 검출한 박스 중에서,
    - 신뢰도가 CONFIDENCE_THRESHOLD 이상이거나
    - 최근 HOLD_SECONDS 이내에 고신뢰였던 박스와 IoU 로 매칭되는 경우
    만 남기고, 마지막에 클래스 간 중복 박스를 한 번 더 제거합니다.
    """
    if result.boxes is None or result.boxes.cls is None:
        recent_boxes = [
            recent for recent in recent_boxes
            if current_time - recent["last_high_confidence_time"] <= HOLD_SECONDS
        ]
        return [], recent_boxes

    boxes_data = result.boxes.data
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    keep_indexes = []
    next_recent_boxes = []
    matched_recent_indexes = set()

    for index in sorted(range(len(boxes_data)), key=lambda i: confidences[i], reverse=True):
        class_id = class_ids[index]
        confidence = confidences[index]
        box = boxes_xyxy[index]

        if class_id not in VEHICLE_NAMES:
            continue

        is_high_confidence = confidence >= CONFIDENCE_THRESHOLD
        matched_recent_index = None

        for recent_index, recent in enumerate(recent_boxes):
            if recent_index in matched_recent_indexes:
                continue
            if recent["class_id"] != class_id:
                continue
            if current_time - recent["last_high_confidence_time"] > HOLD_SECONDS:
                continue
            if box_iou(recent["box"], box) < MATCH_IOU_THRESHOLD:
                continue

            matched_recent_index = recent_index
            matched_recent_indexes.add(recent_index)
            break

        if is_high_confidence or matched_recent_index is not None:
            keep_indexes.append(index)

        if is_high_confidence:
            next_recent_boxes.append({
                "box": box,
                "class_id": class_id,
                "last_high_confidence_time": current_time,
            })
        elif matched_recent_index is not None:
            recent = recent_boxes[matched_recent_index]
            next_recent_boxes.append({
                "box": box,
                "class_id": class_id,
                "last_high_confidence_time": recent["last_high_confidence_time"],
            })

    for recent_index, recent in enumerate(recent_boxes):
        if recent_index in matched_recent_indexes:
            continue
        if current_time - recent["last_high_confidence_time"] <= HOLD_SECONDS:
            next_recent_boxes.append(recent)

    # 클래스 간 중복 박스 하드 필터
    keep_indexes = remove_cross_class_duplicate_boxes(
        keep_indexes,
        boxes_xyxy,
        class_ids,
        confidences,
    )

    # keep_indexes 로 박스를 갱신 (data 7열이면 track id 도 함께 유지됨)
    result.update(boxes=boxes_data[keep_indexes])
    return keep_indexes, next_recent_boxes


def count_detections_by_class(result):
    counts = {class_id: 0 for class_id in VEHICLE_NAMES}
    if result.boxes is None or result.boxes.cls is None:
        return counts

    for class_id in result.boxes.cls.cpu().numpy().astype(int):
        if class_id in counts:
            counts[class_id] += 1

    return counts


def cleanup_stationary_state(state, now):
    """오랫동안 보이지 않은 추적 상태를 제거합니다."""
    stale_ids = [
        track_id
        for track_id, info in state.items()
        if now - info["last_seen"] > TRACK_TIMEOUT_SECONDS
    ]
    for track_id in stale_ids:
        del state[track_id]


def update_vehicle_states(result, state, now, no_parking_zone):
    """
    추적 ID별로
      - 같은 자리에 STATIONARY_LIMIT_SECONDS 이상 멈춤 → emergency
      - 주정차 금지구역 안에 ILLEGAL_LIMIT_SECONDS 이상 머묾 → illegal
    을 판정합니다.
    반환값: (emergencies, illegals)  각 원소: {"id","box","duration","class_id","captured"}
    """
    emergencies = []
    illegals = []
    boxes = result.boxes

    if boxes is None or boxes.id is None:
        cleanup_stationary_state(state, now)
        return emergencies, illegals

    ids = boxes.id.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for track_id, box, class_id in zip(ids, xyxy, clss):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        info = state.get(track_id)
        if info is None:
            # 처음 보는 차량 → 추적 시작
            info = {
                "ref_center": center,
                "since": now,          # 현재 위치에 멈춰 있기 시작한 시각
                "captured": False,     # emergency 캡쳐 여부
                "last_seen": now,
                "class_id": class_id,
                "zone_since": None,    # 금지구역에 들어온 시각
                "illegal_captured": False,
            }
            state[track_id] = info
            is_new = True
        else:
            is_new = False

        # ── emergency(정지) 판정 ──────────────────────────────
        if not is_new:
            dx = center[0] - info["ref_center"][0]
            dy = center[1] - info["ref_center"][1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > MOVEMENT_THRESHOLD_PX:
                info["ref_center"] = center
                info["since"] = now
                info["captured"] = False

        info["last_seen"] = now
        info["class_id"] = class_id

        stationary_seconds = now - info["since"]
        if stationary_seconds >= STATIONARY_LIMIT_SECONDS:
            emergencies.append(
                {
                    "id": int(track_id),
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "duration": stationary_seconds,
                    "class_id": int(class_id),
                    "captured": info["captured"],
                }
            )

        # ── illegal(주정차 금지구역) 판정 ──────────────────────
        if no_parking_zone is not None and center_in_box(center, no_parking_zone):
            if info["zone_since"] is None:
                info["zone_since"] = now
                info["illegal_captured"] = False

            zone_seconds = now - info["zone_since"]
            if zone_seconds >= ILLEGAL_LIMIT_SECONDS:
                illegals.append(
                    {
                        "id": int(track_id),
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "duration": zone_seconds,
                        "class_id": int(class_id),
                        "captured": info["illegal_captured"],
                    }
                )
        else:
            # 구역을 벗어났거나 구역이 없으면 타이머 리셋
            info["zone_since"] = None
            info["illegal_captured"] = False

    cleanup_stationary_state(state, now)
    return emergencies, illegals


def save_capture(frame, item, category):
    """emergency / illegal 발생 프레임을 captures 폴더에 저장합니다."""
    CAPTURE_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    class_name = VEHICLE_NAMES.get(item["class_id"], "vehicle")
    filename = CAPTURE_DIR / f"{category}_{class_name}_id{item['id']}_{stamp}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"[CAPTURE] {category} 저장: {filename}")


def make_mouse_callback(draw_state):
    """드래그로 주정차 금지구역(사각형)을 설정하는 마우스 콜백."""
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            draw_state["drawing"] = True
            draw_state["start"] = (x, y)
            draw_state["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and draw_state["drawing"]:
            draw_state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and draw_state["drawing"]:
            draw_state["drawing"] = False
            draw_state["current"] = (x, y)
            sx, sy = draw_state["start"]
            zx1, zx2 = sorted((sx, x))
            zy1, zy2 = sorted((sy, y))
            # 너무 작은 영역은 무시
            if (zx2 - zx1) > 5 and (zy2 - zy1) > 5:
                draw_state["zone"] = (zx1, zy1, zx2, zy2)
    return on_mouse


def main():

    if not ENGINE_PATH.is_file():
        raise FileNotFoundError(f"TensorRT 엔진을 찾을 수 없습니다: {ENGINE_PATH}")

    # 5종 차량 데이터로 학습하고 Jetson에서 생성한 TensorRT 엔진을 로드합니다.
    model = YOLO(str(ENGINE_PATH), task="detect")
    # print("model.names =", model.names)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: index={CAMERA_INDEX}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    # 드래그로 금지구역을 설정하기 위한 상태 + 마우스 콜백 등록
    draw_state = {"drawing": False, "start": None, "current": None, "zone": None}
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(draw_state))

    previous_time = time.perf_counter()

    # 신뢰도 홀드 필터용 직전 박스 기록
    recent_boxes = []
    # 추적 ID별 정지/금지구역 상태
    stationary_state = {}

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            # 낮은 conf 로 추적: 일단 많이 받아온 뒤 홀드/하드 필터로 거른다.
            results = model.track(
                source=frame,
                imgsz=MODEL_IMAGE_SIZE,
                conf=LOW_CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=0,
                persist=True,            # 프레임 간 추적 상태 유지
                tracker="bytetrack.yaml",
                verbose=False,
            )

            result = results[0]
            result.names = VEHICLE_NAMES

            current_time = time.perf_counter()

            # 신뢰도 홀드 + 클래스 간 중복 제거 필터 (추적 ID 보존됨)
            _, recent_boxes = filter_boxes_with_confidence_hold(
                result,
                recent_boxes,
                current_time,
            )

            display_frame = result.plot()

            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            no_parking_zone = draw_state["zone"]

            # ── 주정차 금지구역 표시 ──────────────────────────
            if no_parking_zone is not None:
                zx1, zy1, zx2, zy2 = no_parking_zone
                cv2.rectangle(display_frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
                cv2.putText(
                    display_frame,
                    "NO PARKING ZONE",
                    (zx1, max(zy1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
            # 드래그 중인 사각형(미리보기)
            if draw_state["drawing"] and draw_state["start"] and draw_state["current"]:
                cv2.rectangle(
                    display_frame,
                    draw_state["start"],
                    draw_state["current"],
                    (0, 255, 255),
                    1,
                )

            # 클래스별 개수 + FPS + 사용법 오버레이
            class_counts = count_detections_by_class(result)
            overlay_lines = [
                f"{VEHICLE_NAMES[class_id]}: {count}"
                for class_id, count in class_counts.items()
            ]
            overlay_lines.append(f"FPS: {fps:.1f}")
            overlay_lines.append("Drag: set zone | c: clear | q: quit")

            for line_index, text in enumerate(overlay_lines):
                cv2.putText(
                    display_frame,
                    text,
                    (10, 30 + line_index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

            # 화면에 검출된 전체 차량 수 (정체 판단 기준)
            total_vehicles = len(result.boxes) if result.boxes is not None else 0

            # ── emergency / illegal 판정 ──────────────────────
            emergencies, illegals = update_vehicle_states(
                result,
                stationary_state,
                current_time,
                no_parking_zone,
            )
            illegal_ids = {item["id"] for item in illegals}

            # ── illegal(주정차 금지) : 검정 박스 + 캡쳐 ─────────
            for item in illegals:
                x1, y1, x2, y2 = item["box"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
                label = f"ILLEGAL id{item['id']} {item['duration']:.0f}s"
                # 검정 박스 위에 라벨이 보이도록 검정 배경 + 흰 글씨
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ly = max(y1 - 8, th + 4)
                cv2.rectangle(display_frame, (x1, ly - th - 4), (x1 + tw, ly + 2), (0, 0, 0), -1)
                cv2.putText(
                    display_frame,
                    label,
                    (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                if not item["captured"]:
                    save_capture(display_frame, item, "illegal")
                    stationary_state[item["id"]]["illegal_captured"] = True

            # ── emergency(정차) : 빨간 박스 + 캡쳐 ──────────────
            # 차량이 많으면 정체 구간으로 보고 emergency 표시/캡쳐 안 함
            if total_vehicles >= CONGESTION_VEHICLE_COUNT:
                cv2.putText(
                    display_frame,
                    f"CONGESTION ({total_vehicles} vehicles)",
                    (10, CAMERA_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
            else:
                for emergency in emergencies:
                    # 금지구역 illegal 로 이미 처리된 차량은 빨간 박스 생략
                    if emergency["id"] in illegal_ids:
                        continue

                    x1, y1, x2, y2 = emergency["box"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"EMERGENCY id{emergency['id']} {emergency['duration']:.0f}s"
                    cv2.putText(
                        display_frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    if not emergency["captured"]:
                        save_capture(display_frame, emergency, "emergency")
                        stationary_state[emergency["id"]]["captured"] = True
            # ──────────────────────────────────────────────────

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # 금지구역 초기화
                draw_state["zone"] = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()