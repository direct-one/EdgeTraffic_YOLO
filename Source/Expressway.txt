from pathlib import Path
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# TensorRT Engine
ENGINE_PATH = Path(__file__).resolve().parent / "vehicle5_yolo11n_e250.engine"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MODEL_IMAGE_SIZE = 640

# ── 검출/신뢰도 홀드 필터 설정 ─────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3          
LOW_CONFIDENCE_THRESHOLD = 0.1      
IOU_THRESHOLD = 0.45                
HOLD_SECONDS = 2                     
MATCH_IOU_THRESHOLD = 0.3           
CROSS_CLASS_IOU_THRESHOLD = 0.5     
# ──────────────────────────────────────────────────────────────

# ── 정차(emergency) 판정 설정 ───────────────────────────────────
STATIONARY_LIMIT_SECONDS = 5.0      

SMOOTHING_WINDOW = 5               
MOVEMENT_THRESHOLD_RATIO = 0.5     
MOVEMENT_THRESHOLD_MIN_PX = 25.0   
RESET_CONFIRM_FRAMES = 4           
TRACK_TIMEOUT_SECONDS = 5.0        
CONGESTION_VEHICLE_COUNT = 8       
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"  
# ──────────────────────────────────────────────────────────────

# ── 영역(zone) 카운트 / 트럭 캡쳐 설정 ──────────────────────────
ZONE_TRACK_TTL_SECONDS = 3.0       
ZONE_COLOR = (0, 255, 255)
ZONE_ALPHA = 0.28
ZONE_GRID_STEP = 18
SCREENSHOT_DIR = Path(__file__).resolve().parent / "truck_screenshots"  # zone에 truck이 들어가면 자동저장 
# ──────────────────────────────────────────────────────────────

# 자동차 Class를 나누기 
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


def box_diagonal(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (w * w + h * h) ** 0.5


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
    """오랫동안 보이지 않은 추적 상태를 제거"""
    stale_ids = [
        track_id
        for track_id, info in state.items()
        if now - info["last_seen"] > TRACK_TIMEOUT_SECONDS
    ]
    for track_id in stale_ids:
        del state[track_id]


def update_vehicle_states(result, state, now):
    """
    카메라 흔들림 내구성을 위해 두 가지를 적용:
    1) 최근 SMOOTHING_WINDOW 프레임의 중심 좌표를 이동평균해서 "현재 위치"로 사용 (고주파 흔들림 제거)
    2) 임계값을 넘는 이동이 RESET_CONFIRM_FRAMES 프레임 연속으로 감지돼야 실제로 리셋
       (흔들림으로 인한 단발성 튐은 무시)

    반환값: emergencies  각 원소: {"id","box","duration","class_id","captured"}
    """
    emergencies = []
    boxes = result.boxes

    if boxes is None or boxes.id is None:
        cleanup_stationary_state(state, now)
        return emergencies

    ids = boxes.id.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for track_id, box, class_id in zip(ids, xyxy, clss):
        x1, y1, x2, y2 = box
        raw_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        diagonal = box_diagonal(box)
        move_threshold = max(MOVEMENT_THRESHOLD_MIN_PX, diagonal * MOVEMENT_THRESHOLD_RATIO)

        info = state.get(track_id)
        if info is None:
            # 처음 보는 차량 → 추적 시작
            info = {
                "center_history": deque([raw_center], maxlen=SMOOTHING_WINDOW),
                "ref_center": raw_center,   # 정지 판정 기준 위치 
                "since": now,               # 현재 위치에 멈춰 있기 시작한 시각
                "captured": False,          # emergency 캡쳐 여부
                "last_seen": now,
                "class_id": class_id,
                "over_threshold_streak": 0,  # 임계값을 넘는 상태가 연속된 프레임 수
            }
            state[track_id] = info
        else:
            info["center_history"].append(raw_center)

        # 이동평균으로 스무딩된 현재 위치
        history = info["center_history"]
        smoothed_cx = sum(p[0] for p in history) / len(history)
        smoothed_cy = sum(p[1] for p in history) / len(history)
        smoothed_center = (smoothed_cx, smoothed_cy)

        # ── emergency(정지) 판정: 스무딩된 좌표 + 연속 프레임 확인 ──
        dx = smoothed_center[0] - info["ref_center"][0]
        dy = smoothed_center[1] - info["ref_center"][1]
        distance = (dx * dx + dy * dy) ** 0.5

        if distance > move_threshold:
            info["over_threshold_streak"] += 1
        else:
            info["over_threshold_streak"] = 0

        if info["over_threshold_streak"] >= RESET_CONFIRM_FRAMES:
            # 흔들림이 아니라 실제로 여러 프레임 연속 이동했다고 판단 → 리셋
            info["ref_center"] = smoothed_center
            info["since"] = now
            info["captured"] = False
            info["over_threshold_streak"] = 0

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

    cleanup_stationary_state(state, now)
    return emergencies


def save_capture(frame, item, category):
    """emergency 발생 프레임을 captures 폴더에 저장."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    class_name = VEHICLE_NAMES.get(item["class_id"], "vehicle")
    filename = CAPTURE_DIR / f"{category}_{class_name}_id{item['id']}_{stamp}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"[CAPTURE] {category} 저장: {filename}")


def on_mouse_click(event, x, y, flags, param):
    """4번 클릭으로 다각형 영역(zone)을 지정하는 마우스 콜백."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    zone_points = param["zone_points"]
    if len(zone_points) >= 4:
        zone_points.clear()

    zone_points.append((x, y))


def get_zone_polygon(zone_points):
    if len(zone_points) != 4:
        return None

    return np.array(zone_points, dtype=np.int32)


def is_center_inside_zone(box, zone_polygon):
    center = box_center(box)
    return cv2.pointPolygonTest(zone_polygon, center, False) >= 0


def update_zone_counts(result, zone_polygon, zone_counts, zone_tracks, current_time):
    """
    1. zone 안에 들어온 차량을 'track_id' 기준으로 카운트합니다.
    2. 같은 track_id가 zone 안에 있는 동안에는 다시 카운트되지 않음,
    3. 추적이 잠깐 끊겨도 ZONE_TRACK_TTL_SECONDS 안에 같은 ID로 복귀하면 이미 카운트된 차량으로 인식 .
    4. 트럭이 처음 카운트되는 바로 그 순간에만 truck_just_entered=True를 반환.

    zone_tracks: { track_id: {"class_id": int, "last_seen": float} }
    """
    truck_just_entered = False

    # TTL이 지난 트랙은 제거 (오래 안 보인 차량 → 새로 들어오면 새 카운트로 처리)
    expired_ids = [
        track_id for track_id, info in zone_tracks.items()
        if current_time - info["last_seen"] > ZONE_TRACK_TTL_SECONDS
    ]
    for track_id in expired_ids:
        del zone_tracks[track_id]

    if (zone_polygon is None or result.boxes is None
            or result.boxes.cls is None or result.boxes.id is None):
        return zone_counts, zone_tracks, truck_just_entered

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = result.boxes.id.cpu().numpy().astype(int)

    for box, class_id, track_id in zip(boxes_xyxy, class_ids, track_ids):
        if class_id not in VEHICLE_NAMES or not is_center_inside_zone(box, zone_polygon):
            continue

        if track_id in zone_tracks:
            # 이미 카운트된 차량 → 위치 정보만 갱신
            zone_tracks[track_id]["last_seen"] = current_time
            continue

        # 새로운 track_id로는 처음 zone에 들어옴 → 새로 카운트
        zone_counts[class_id] += 1
        zone_tracks[track_id] = {"class_id": class_id, "last_seen": current_time}
        if VEHICLE_NAMES[class_id] == "truck":
            truck_just_entered = True

    return zone_counts, zone_tracks, truck_just_entered


def save_truck_screenshot(frame, current_time, truck_count):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
    milliseconds = int((current_time - int(current_time)) * 1000)
    filename = f"truck_{truck_count:04d}_{timestamp}_{milliseconds:03d}.jpg"
    cv2.imwrite(str(SCREENSHOT_DIR / filename), frame)
    print(f"[CAPTURE] truck zone 진입 저장: {filename}")


def draw_zone(frame, zone_points):
    """4점 미만이면 점/선만, 4점이 모이면 반투명 격자 채움 + 외곽선으로 zone을 표시합니다."""
    if not zone_points:
        return

    points = np.array(zone_points, dtype=np.int32)
    if len(points) < 4:
        for point in points:
            cv2.circle(frame, tuple(point), 4, ZONE_COLOR, -1)
        if len(points) > 1:
            cv2.polylines(frame, [points], False, ZONE_COLOR, 1, lineType=cv2.LINE_AA)
        return

    polygon = points.reshape((-1, 1, 2))
    overlay = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    x, y, w, h = cv2.boundingRect(polygon)
    for grid_x in range(x, x + w + 1, ZONE_GRID_STEP):
        cv2.line(overlay, (grid_x, y), (grid_x, y + h), ZONE_COLOR, 1, lineType=cv2.LINE_AA)
    for grid_y in range(y, y + h + 1, ZONE_GRID_STEP):
        cv2.line(overlay, (x, grid_y), (x + w, grid_y), ZONE_COLOR, 1, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1.0 - ZONE_ALPHA, 0)
    frame[mask > 0] = blended[mask > 0]
    cv2.polylines(frame, [polygon], True, ZONE_COLOR, 1, lineType=cv2.LINE_AA)


def draw_zone_counts(frame, zone_counts):
    title_y = 220
    cv2.putText(
        frame,
        "Zone",
        (10, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ZONE_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )

    for line_index, class_id in enumerate(VEHICLE_NAMES):
        text = f"{VEHICLE_NAMES[class_id]}: {zone_counts[class_id]}"
        cv2.putText(
            frame,
            text,
            (10, title_y + 30 + line_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            ZONE_COLOR,
            2,
            lineType=cv2.LINE_AA,
        )


def main():
    if not ENGINE_PATH.is_file():
        raise FileNotFoundError(f"TensorRT 엔진을 찾을 수 없습니다: {ENGINE_PATH}")

    # 5종 차량 데이터로 학습하고 Jetson에서 생성한 TensorRT 엔진을 로드.
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

    # 4번 클릭으로 zone(다각형) 을 설정하기 위한 상태 + 마우스 콜백 등록
    zone_points = []
    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click, {"zone_points": zone_points})

    previous_time = time.perf_counter()

    # 신뢰도 홀드 필터용 직전 박스 기록
    recent_boxes = []
    # 추적 ID별 정지(emergency) 상태
    stationary_state = {}
    # zone 누적 카운트 + zone 안 트랙 기록 (track_id 기준 딕셔너리)
    zone_counts = {class_id: 0 for class_id in VEHICLE_NAMES}
    zone_tracks = {}

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

            # ── zone(영역) 카운트 + 표시 ──────────────────────
            zone_polygon = get_zone_polygon(zone_points)
            zone_counts, zone_tracks, truck_just_entered = update_zone_counts(
                result,
                zone_polygon,
                zone_counts,
                zone_tracks,
                current_time,
            )
            draw_zone(display_frame, zone_points)
            draw_zone_counts(display_frame, zone_counts)
            if truck_just_entered:
                save_truck_screenshot(display_frame, current_time, zone_counts[0])

            # 클래스별 개수 + FPS + 사용법 오버레이
            class_counts = count_detections_by_class(result)
            overlay_lines = [
                f"{VEHICLE_NAMES[class_id]}: {count}"
                for class_id, count in class_counts.items()
            ]
            overlay_lines.append(f"FPS: {fps:.1f}")
            overlay_lines.append("Click x4: set zone | q: quit")

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

            # ── emergency(정차) 판정 ───────────────────────────
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
                emergencies = update_vehicle_states(result, stationary_state, current_time)
                for emergency in emergencies:
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

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()