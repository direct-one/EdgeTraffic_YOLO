# 🚗 Vehicle Detection System

YOLO11 + TensorRT + ByteTrack 기반의 **실시간 차량 검출 및 모니터링 시스템**입니다.  
카메라 영상에서 5종 차량(truck, trailer, bus, car, motorcycle)을 검출·추적하고, 정차 감지 및 주정차 금지구역 위반을 자동으로 판별합니다.

---

## 📋 목차
- [✨ 주요 기능](#-주요-기능)
- [💻 시스템 요구사항](#-시스템-요구사항)
- [📦 설치](#-설치)
- [🚀 실행](#-실행)
- [🖱️ 사용법](#-사용법)
- [⚙️ 설정값](#-설정값)
- [📁 코드 구조](#-코드-구조)
- [📸 출력 파일](#-출력-파일)
- [🛠️ 기술 스택](#-기술-스택)
- [🏗️ 아키텍처](#-아키텍처)
- [🙏 참고](#-참고)

---

## ✨ 주요 기능

이 프로젝트는 두 가지 버전으로 구성되어 있습니다.

### v1 — Zone 카운트 버전
| 기능 | 설명 |
| :--- | :--- |
| **실시간 차량 검출** | 5종 차량을 YOLO11n TensorRT 엔진으로 실시간 검출 |
| **다중 객체 추적** | ByteTrack 알고리즘으로 프레임 간 차량 ID 유지 |
| **신뢰도 홀드 필터** | 고신뢰 검출이 사라져도 2초간 유지하여 깜빡임 방지 |
| **클래스 간 중복 제거** | 같은 차량이 car+truck 등으로 이중 검출되는 것을 방지 |
| **정차(Emergency) 감지** | 5초 이상 정지한 차량을 감지하고 자동 캡쳐 |
| **카메라 흔들림 내구성**| 이동평균(5프레임) + 연속 프레임 확인(4프레임) 이중 안전장치 |
| **Zone 영역 카운트** | 마우스 4클릭으로 다각형 영역 설정 → 진입 차량을 클래스별 누적 카운트 |
| **트럭 자동 캡쳐** | 트럭이 Zone에 진입하면 자동 스크린샷 저장 |
| **정체 구간 판단** | 8대 이상 검출 시 정체로 판단하여 Emergency 표시 비활성화 |

### v2 — 주정차 금지구역 버전
| 기능 | 설명 |
| :--- | :--- |
| **실시간 차량 검출** | v1과 동일 |
| **다중 객체 추적** | v1과 동일 |
| **신뢰도 홀드 필터** | v1과 동일 |
| **클래스 간 중복 제거** | v1과 동일 |
| **정차(Emergency) 감지** | 5초 이상 정지 감지 (고정 20px 임계값) |
| **주정차 금지구역(Illegal) 감지** | 드래그로 설정한 사각형 영역에 10초 이상 머무는 차량 감지 |
| **자동 캡쳐** | Emergency / Illegal 발생 시 해당 프레임을 자동 저장 |
| **정체 구간 판단** | v1과 동일 |

### v1 vs v2 비교
| 항목 | v1 (Zone 카운트) | v2 (주정차 금지구역) |
| :--- | :--- | :--- |
| **영역 입력 방식** | 4번 클릭 → 다각형 | 드래그 → 사각형 |
| **흔들림 대응** | 이동평균 + 연속 프레임 (강건) | 고정 20px (단순) |
| **Illegal 판정** | ❌ | ✅ (10초 타이머) |
| **Zone 카운트** | ✅ (track_id 기반) | ❌ |
| **트럭 자동 캡쳐**| ✅ | ❌ |

---

## 💻 시스템 요구사항

### 하드웨어
* **NVIDIA Jetson** (Nano / Xavier NX / Orin 등)
* **CUDA 지원 GPU**
* **USB 또는 CSI 카메라**

### 소프트웨어
* Python 3.8+
* JetPack SDK (CUDA, cuDNN, TensorRT 포함)
* OpenCV (with CUDA support 권장)
* Ultralytics YOLO

---

## 📦 설치

### 1. 의존성 설치
```bash
pip install ultralytics opencv-python numpy
```

### 2. TensorRT 엔진 준비
학습된 YOLO11n 모델(`.pt`)을 Jetson에서 TensorRT 엔진으로 변환합니다:
```bash
yolo export model=vehicle5_yolo11n_e250.pt format=engine device=0 imgsz=640
```
> ⚠️ TensorRT 엔진은 **생성한 장치에서만** 사용 가능합니다. 다른 Jetson 모델로 옮기면 재변환이 필요합니다.

### 3. 파일 배치
```text
프로젝트/
├── vehicle_detect_v1.py              # v1: Zone 카운트 버전
├── vehicle_detect_v2.py              # v2: 주정차 금지구역 버전
├── vehicle5_yolo11n_e250.engine      # TensorRT 엔진 (동일 폴더에 배치)
└── bytetrack.yaml                    # ByteTrack 설정 파일
```

---

## 🚀 실행

```bash
# v1 실행 (Zone 카운트)
python vehicle_detect_v1.py

# v2 실행 (주정차 금지구역)
python vehicle_detect_v2.py
```

---

## 🖱️ 사용법

### v1 — Zone 카운트 버전
| 조작 | 동작 |
| :--- | :--- |
| **화면 4번 클릭** | 다각형 Zone 영역 설정 (진입 차량 카운트 시작) |
| **다시 4번 클릭** | 기존 Zone 초기화 후 새 Zone 설정 |
| **`q` 키** | 프로그램 종료 |

### v2 — 주정차 금지구역 버전
| 조작 | 동작 |
| :--- | :--- |
| **마우스 드래그** | 주정차 금지구역(사각형) 설정 |
| **`c` 키** | 금지구역 초기화 |
| **`q` 키** | 프로그램 종료 |

### 화면 표시 색상
| 색상 | 의미 |
| :--- | :--- |
| 🟢 초록색 텍스트 | 클래스별 검출 수, FPS, 사용법 안내 |
| 🔴 빨간 박스 | Emergency — 5초 이상 정차 |
| ⬛ 검정 박스 + 흰 글씨 | Illegal — 금지구역 10초 이상 체류 (v2) |
| 🟡 노란색 사각형/격자 | Zone 영역 (v1) 또는 금지구역 (v2) |
| 🟠 주황색 텍스트 | Congestion — 정체 구간 (8대 이상) |

---

## ⚙️ 설정값

### 검출 / 필터
| 상수 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD` | 0.3 | 이 이상이면 "확실한" 검출 |
| `LOW_CONFIDENCE_THRESHOLD` | 0.1 | 추적 단계에서 받아올 최소 신뢰도 |
| `IOU_THRESHOLD` | 0.45 | NMS IoU 임계값 |
| `HOLD_SECONDS` | 2 | 고신뢰 검출이 사라져도 유지하는 시간(초) |
| `MATCH_IOU_THRESHOLD` | 0.3 | 이전 박스와 매칭할 IoU 임계값 |
| `CROSS_CLASS_IOU_THRESHOLD` | 0.5 | 클래스 간 중복으로 판단하는 IoU |

### 정차 / 금지구역 판정
| 상수 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `STATIONARY_LIMIT_SECONDS` | 5.0 | 이 시간 이상 정지 → Emergency |
| `ILLEGAL_LIMIT_SECONDS` | 10.0 | 금지구역 체류 시간 → Illegal (v2) |
| `MOVEMENT_THRESHOLD_PX` | 20.0 | 정지 판정 이동 임계값 (v2) |
| `MOVEMENT_THRESHOLD_RATIO` | 0.5 | 박스 대각선 대비 이동 비율 (v1) |
| `SMOOTHING_WINDOW` | 5 | 이동평균 프레임 수 (v1) |
| `RESET_CONFIRM_FRAMES` | 4 | 이동 확인 연속 프레임 수 (v1) |
| `CONGESTION_VEHICLE_COUNT` | 8 | 이 수 이상이면 정체로 판단 |
| `TRACK_TIMEOUT_SECONDS` | 5.0 | 추적 끊김 후 상태 삭제 시간 |

### 카메라 / 모델
| 상수 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `CAMERA_INDEX` | 0 | 카메라 장치 번호 |
| `CAMERA_WIDTH` | 640 | 카메라 해상도 (너비) |
| `CAMERA_HEIGHT`| 480 | 카메라 해상도 (높이) |
| `MODEL_IMAGE_SIZE` | 640 | YOLO 추론 입력 크기 |

---

## 📁 코드 구조

### 공통 함수
| 함수 | 역할 |
| :--- | :--- |
| `box_iou(a, b)` | 두 박스의 IoU(겹침 비율) 계산 |
| `box_center(box)` | 박스 중심점 반환 |
| `center_in_box(center, box)` | 점이 박스 안에 있는지 판정 |
| `is_cross_class_duplicate(...)` | 다른 클래스 간 같은 위치 중복 판정 |
| `remove_cross_class_duplicate_boxes(...)` | 중복 박스 중 저신뢰 제거 (Greedy) |
| `filter_boxes_with_confidence_hold(...)` | **핵심 필터** — 홀드 + 중복 제거 |
| `count_detections_by_class(result)` | 클래스별 검출 수 집계 |
| `cleanup_stationary_state(state, now)` | 오래된 추적 상태 메모리 정리 |
| `update_vehicle_states(...)` | **정차/금지구역 판정** |
| `save_capture(frame, item, category)` | 캡쳐 프레임 JPEG 저장 |
| `main()` | 카메라 초기화 → 메인 루프 → 리소스 해제 |

### v1 전용 함수
| 함수 | 역할 |
| :--- | :--- |
| `box_diagonal(box)` | 박스 대각선 길이 계산 |
| `on_mouse_click(...)` | 4번 클릭으로 Zone 다각형 설정 |
| `get_zone_polygon(points)` | 4점 → NumPy 폴리곤 배열 변환 |
| `is_center_inside_zone(box, polygon)` | 박스 중심이 Zone 안에 있는지 판정 |
| `update_zone_counts(...)` | track_id 기반 Zone 진입 카운트 |
| `save_truck_screenshot(...)` | 트럭 Zone 진입 시 자동 캡쳐 |
| `draw_zone(frame, points)` | 반투명 격자 Zone 시각화 |
| `draw_zone_counts(frame, counts)` | Zone 클래스별 카운트 표시 |

### v2 전용 함수
| 함수 | 역할 |
| :--- | :--- |
| `make_mouse_callback(draw_state)` | 드래그로 금지구역 설정하는 클로저 콜백 |

---

## 📸 출력 파일

### 저장 경로
```text
프로젝트/
├── captures/                          # Emergency / Illegal 자동 캡쳐
│   ├── emergency_car_id12_20260630_165030_412.jpg
│   └── illegal_truck_id7_20260630_165040_831.jpg
└── truck_screenshots/                 # v1: 트럭 Zone 진입 캡쳐
    └── truck_0003_20260630_165050_123.jpg
```

### 파일명 형식
| 유형 | 형식 | 예시 |
| :--- | :--- | :--- |
| Emergency | `emergency_{class}_id{id}_{timestamp}.jpg` | `emergency_car_id12_20260630_165030_412.jpg` |
| Illegal | `illegal_{class}_id{id}_{timestamp}.jpg` | `illegal_truck_id7_20260630_165040_831.jpg` |
| Truck Zone 진입 | `truck_{count}_{timestamp}_{ms}.jpg` | `truck_0003_20260630_165050_123.jpg` |
> 📌 각 이벤트당 **최초 1회만** 캡쳐됩니다 (중복 저장 방지).

---

## 🛠️ 기술 스택

```text
┌─────────────────────────────────────────────────┐
│  하드웨어     NVIDIA Jetson + CUDA GPU + 카메라    │
├─────────────────────────────────────────────────┤
│  AI 엔진     YOLO11n → TensorRT 추론 최적화       │
│              ByteTrack 다중 객체 추적 (MOT)        │
├─────────────────────────────────────────────────┤
│  후처리      신뢰도 홀드 필터 (깜빡임 방지)          │
│  (Python)    클래스 간 중복 제거 (Greedy NMS)       │
│              정차/금지구역 판정 (상태 머신)           │
├─────────────────────────────────────────────────┤
│  라이브러리   OpenCV (영상 I/O, 시각화)             │
│              Ultralytics (YOLO 프레임워크)          │
│              NumPy (배열 연산)                      │
├─────────────────────────────────────────────────┤
│  출력        실시간 화면 표시 + 이벤트 자동 캡쳐      │
└─────────────────────────────────────────────────┘
```

---

## 🏗️ 아키텍처

```text
카메라 입력 (640×480)
    │
    ▼
YOLO11n TensorRT 추론 (conf=0.1, GPU)
    │
    ▼
ByteTrack 다중 객체 추적 (persist=True)
    │
    ▼
┌──────────────────────────────────┐
│  신뢰도 홀드 필터                  │
│  ├─ 고신뢰(≥0.3) → 즉시 통과      │
│  ├─ 저신뢰 + 이전 매칭 → 2초 유지  │
│  └─ 클래스 간 중복 제거            │
└──────────────────────────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌──────────┐   ┌──────────────┐
│ 정차 판정  │   │ Zone/금지구역  │
│ Emergency │   │   카운트/판정  │
│ (5초 정지) │   │  (v1 or v2)  │
└──────────┘   └──────────────┘
    │                  │
    ▼                  ▼
┌──────────────────────────────────┐
│  화면 출력 + 자동 캡쳐 저장         │
│  ├─ 🔴 Emergency 빨간 박스        │
│  ├─ ⬛ Illegal 검정 박스 (v2)      │
│  ├─ 🟡 Zone/금지구역 표시          │
│  └─ 🟠 Congestion 정체 알림       │
└──────────────────────────────────┘
```

---


## 🙏 참고
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
