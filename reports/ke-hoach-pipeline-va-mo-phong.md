# Kế hoạch triển khai chính thức — Pipeline video UAV và mô phỏng SUMO

**Trạng thái:** Roadmap snapshot ngày 2026-08-27; WP0/WP1 đã được triển khai phần lớn, các WP SUMO chưa có trong runtime. Hiện trạng code xem `reports/kien-truc-pipeline.md`.
**Ngày khóa định hướng:** 2026-08-27  
**Phạm vi:** Hệ thống phân tích giao thông từ video UAV và mô phỏng ảnh hưởng của sai số tri giác  
**Tài liệu trước:** `ke-hoach-cai-tien.md` (lịch sử hình thành ý tưởng) đã được dọn bỏ cùng các tài liệu nghiên cứu/benchmark cũ; đây là tài liệu triển khai duy nhất còn hiệu lực.

---

## 1. Tóm tắt quyết định

Dự án chuyển từ giai đoạn tìm kiếm cải tiến riêng lẻ cho detector/tracker sang giai đoạn xây dựng và đánh giá một hệ thống hoàn chỉnh.

Hai deliverable bắt buộc, theo đúng thứ tự, là:

1. **Pipeline video UAV chạy end-to-end:** phát hiện → tracking → đếm/analytics → đánh giá độ tin cậy → evidence → VLM/LLM → báo cáo.
2. **Mô phỏng SUMO:** tạo traffic ground truth có kiểm soát, chèn sai số tri giác đo được từ pipeline thật, rồi so sánh ảnh hưởng lên analytics và điều khiển giao thông.

SUMO không còn là một ghi chú “có thể làm sau”. Nó là nhánh thứ hai của sản phẩm nghiên cứu, được triển khai sau khi interface và output của pipeline video đã ổn định.

CityFlow, fine-tuning, quantization và edge deployment vẫn là hướng mở rộng, không phải việc đang hoạt động trong kế hoạch này.

---

## 2. Vì sao chuyển sang hướng này

### 2.1. Pipeline hiện tại đã đủ nền tảng để chuyển pha

Codebase đã có các thành phần chính:

- detector cho video UAV;
- BoT-SORT/ByteTrack và benchmark tracking;
- ROI, counting line, trajectory và traffic analytics;
- state machine, hysteresis và cảnh báo;
- evidence keyframe/clip có provenance;
- VLM và LLM pretrained với contract JSON;
- output `annotated.mp4`, `tracks.csv`, `analytics.csv`, `events.jsonl`, `evidence.jsonl`, `summary.json` và `run.json`;
- Bộ unit test hiện tại được kiểm tra bằng `python -m unittest discover -s tests -v`; không khóa số lượng test trong kế hoạch vì suite tiếp tục thay đổi.

Tracking đã được xem là **đóng băng ở mức sản phẩm**:

- profile chất lượng: cấu hình ReID đang dùng ở run73;
- profile nhanh: cấu hình không ReID được tinh chỉnh ở run80;
- proxy fragmentation ủng hộ profile nhanh, nhưng đánh giá trực quan tại vùng che khuất ủng hộ profile ReID;
- không tiếp tục tracker sweep nếu không có regression cụ thể.

### 2.2. Các pilot trước đã cung cấp bài học, không phải backlog phải tiếp tục

Các hướng NWD, P2 head, SAHI bắt buộc, GMC-as-fix, stillness scalar, TVLR oracle và nhiều tracker sweep đã giúp xác định giới hạn của dữ liệu/phương pháp. Tuy nhiên, chúng cũng cho thấy một rủi ro lặp lại:

1. thấy một failure case;
2. đề xuất một module;
3. chạy pilot;
4. kết quả âm hoặc proxy không khớp chất lượng thật;
5. mở thêm một nhánh mới trước khi hệ thống chính hoàn tất.

Kế hoạch mới không phủ nhận giá trị của thử nghiệm. Nó chỉ yêu cầu mọi thử nghiệm mới phải phục vụ một acceptance gate hoặc một câu hỏi nghiên cứu đã khóa trước.

### 2.3. Video nguồn không đủ để suy luận luật địa phương

Video UAV có thể đến từ nhiều địa điểm, độ cao, góc quay và thời điểm khác nhau. Người sử dụng, kể cả tác giả dự án, có thể không biết chắc:

- tên hướng địa lý;
- quy định theo làn;
- có được rẽ phải khi đèn đỏ hay không;
- pha đèn tại thời điểm quay;
- hình học chính xác ngoài phần nhìn thấy trong ảnh.

Vì vậy hệ thống không được đoán luật từ hình ảnh. Scene manifest ban đầu được thu nhỏ thành **measurement manifest**, chỉ mô tả hình học đo lường do người dùng trực tiếp đặt trên ảnh.

### 2.4. Mô phỏng trả lời câu hỏi mà video thật không thể trả lời một mình

Video thật cho biết hệ thống CV sai như thế nào, nhưng không cung cấp ground truth đầy đủ cho mọi thời điểm và không cho phép thay đổi demand hay tín hiệu đèn có kiểm soát.

SUMO cung cấp:

- trạng thái phương tiện sạch theo từng bước thời gian;
- network, route, demand và vehicle type có kiểm soát;
- traffic-light program;
- nhiều seed và mức demand tái lập được;
- Python TraCI để đọc trạng thái và điều khiển đèn.

Do đó dự án có thể đo:

> Sai số detection/tracking/counting quan sát trên video UAV ảnh hưởng thế nào đến analytics và điều khiển, và một cơ chế reliability-aware có giảm tác động đó hay không?

SUMO được dùng để đánh giá tác động downstream của sai số tri giác, **không** được dùng để tuyên bố chất lượng detector trên ảnh.

---

## 3. Câu hỏi nghiên cứu

### RQ1 — Hệ thống video UAV

Làm thế nào xây dựng một pipeline phát hiện, tracking, đếm và mô tả giao thông có thể kiểm toán, trong đó số liệu tất định được tách khỏi nhận xét thị giác không chắc chắn?

### RQ2 — Failure awareness

Làm thế nào tránh trường hợp detector im lặng hoặc mất recall bị diễn giải sai thành “đường thông thoáng”?

### RQ3 — Mô phỏng downstream impact

Sai số tri giác đo được từ pipeline thật làm thay đổi waiting time, queue length, throughput và travel time của một traffic controller trong SUMO bao nhiêu?

### RQ4 — Reliability-aware fallback

Khi observation không đáng tin, việc quay về một fixed-time safe plan có ổn định hơn việc tiếp tục điều khiển thích nghi từ dữ liệu nhiễu hay không?

RQ4 là giả thuyết cần kiểm chứng, không phải kết luận được giả định trước. Kết quả bằng 0 hoặc âm vẫn là kết quả hợp lệ nếu protocol được giữ nguyên.

---

## 4. Mục tiêu và tiêu chí thành công

### 4.1. Mục tiêu hệ thống thật

- Chạy end-to-end trên ít nhất hai video UAV có đặc tính khác nhau.
- Sinh đầy đủ output có cấu trúc và video annotate.
- Tracking và counting dùng cấu hình đã khóa.
- Detection silence không tự động được gọi là `NORMAL`.
- Thiếu measurement manifest không làm pipeline lỗi.
- VLM chỉ đưa ra nhận xét thị giác có evidence reference.
- LLM không được thay đổi hoặc tự sinh số liệu analytics.
- Mỗi run ghi đủ config, source, model, tracker và provenance hash cần thiết.

### 4.2. Mục tiêu mô phỏng

- Có một scenario SUMO tổng hợp chạy được từ config và seed cố định.
- SUMO adapter xuất cùng schema quan sát với pipeline video.
- Error injector tái lập được từ seed và cấu hình.
- Chạy được ba chế độ: `oracle`, `perception_noisy`, `reliability_aware`.
- Báo cáo đầy đủ chỉ số giao thông và mức fallback.
- So sánh chỉ dùng cùng network, demand, seed và controller parameters.

### 4.3. Tiêu chí thành công không phụ thuộc kết quả dương

Dự án được coi là thành công khi:

- hệ thống chạy được;
- phép đo đúng với định nghĩa;
- protocol tái lập được;
- claim không vượt quá bằng chứng;
- kết quả dương, âm hoặc trung tính đều được báo cáo.

Không đặt điều kiện “controller reliability-aware bắt buộc phải thắng” hay “mAP bắt buộc phải tăng” sau khi đã bắt đầu đánh giá.

---

## 5. Phạm vi đang hoạt động

### 5.1. Có trong scope

- Pipeline video UAV end-to-end.
- Detector/tracker checkpoint đã chọn.
- Counting theo frame, ROI và counting line.
- `perception_status` và traffic state `UNKNOWN` khi bằng chứng không đủ.
- Measurement manifest tối thiểu, tùy chọn.
- Multi-view evidence cho VLM: full frame + ROI/crop.
- LLM report tiếng Việt từ structured events và validated VLM observations.
- Schema quan sát chung cho video và SUMO.
- SUMO adapter.
- Scenario ngã tư tổng hợp.
- Perception error injector.
- Fixed-time, adaptive và reliability-aware controller cơ bản.
- Oracle/noisy/reliability-aware evaluation.

### 5.2. Ngoài scope hiện tại

- Suy luận luật giao thông địa phương từ video.
- Phát hiện đi sai làn, rẽ sai luật hoặc vượt đèn đỏ.
- Digital twin của địa điểm quay thật.
- Dựng lại GPS/IMU, camera calibration hoặc BEV khi không có metadata.
- Render ảnh UAV tổng hợp để benchmark detector.
- Chạy detector trực tiếp trong SUMO.
- Triển khai đồng thời SUMO và CityFlow.
- Reinforcement learning hoặc multi-agent RL.
- Fine-tune VLM/LLM.
- Quantization và edge deployment.
- Mở lại NWD/P2/SAHI/TVLR/ReID/tracker tuning nếu không có regression tái hiện được.

### 5.3. Future work đã xác định

- CityFlow adapter dùng lại schema chung.
- RL/multi-agent traffic-signal control.
- Digital twin khi có map, signal plan, demand và metadata đáng tin.
- Quantization detector/VLM/LLM.
- NPU/edge deployment.
- Fine-tuning có dataset và evaluation protocol phù hợp.

---

## 6. Kiến trúc hai nhánh

```text
                           NHÁNH VIDEO THẬT

Video UAV
  → Decode / timestamp
  → Detector
  → Tracker
  → VideoObservationAdapter
                         ┐
                         │
                         ▼
                 TrafficObservation
                         │
          ┌──────────────┼───────────────┐
          ▼              ▼               ▼
      Analytics      Reliability      Evidence
          │              gate             │
          └──────────────┼───────────────┘
                         ▼
                 Events / Counting
                         │
                 ┌───────┴────────┐
                 ▼                ▼
                VLM              LLM
          (video evidence)   (structured report)


                           NHÁNH MÔ PHỎNG

SUMO network + demand + seed
  → SUMO ground truth
  → SumoObservationAdapter ───────────────┐
                                          ├→ TrafficObservation
  → PerceptionErrorInjector ──────────────┘
          │
          ├→ oracle
          ├→ perception_noisy
          └→ reliability_aware
                    │
                    ▼
             Controller / Metrics
```

VLM chỉ thuộc nhánh có ảnh/video. LLM có thể mô tả structured outputs của cả hai nhánh, nhưng báo cáo mô phỏng phải ghi rõ `source=simulation` và không chứa visual claims.

---

## 7. Schema quan sát chung

Schema tối thiểu cần đủ cho analytics và mô phỏng, nhưng không phụ thuộc bbox hay đơn vị riêng của một backend.

```json
{
  "schema_version": 1,
  "source": "video | sumo_clean | sumo_noisy",
  "run_id": "...",
  "timestamp_s": 12.5,
  "frame_or_step": 375,
  "perception_status": {
    "state": "reliable | degraded | detection_silence | unavailable",
    "reason_codes": [],
    "confidence": null
  },
  "objects": [
    {
      "object_id": "42",
      "class_name": "motorcycle",
      "region_id": "approach_a",
      "position": [0.41, 0.63],
      "speed": 4.2,
      "confidence": 0.78,
      "observation_kind": "tracked | simulator_ground_truth"
    }
  ]
}
```

Quy ước cần khóa trước khi code:

- đơn vị của `speed` và `position` phải được khai báo trong metadata;
- video có thể dùng normalized image coordinates và px/s;
- SUMO dùng lane/road coordinates và m/s;
- analytics chỉ so sánh các trường có cùng semantic, không ngầm coi px/s là m/s;
- `object_id` trong video là tracker ID, không được gọi là physical identity ground truth;
- `sumo_clean` dùng vehicle ID thật của simulator;
- mọi transformation phải lưu version và config hash.

---

## 8. Measurement manifest tối thiểu

Manifest không chứa luật giao thông. Nó chỉ mô tả vùng đo trên video.

```yaml
schema_version: 1
scene_id: video_7938

camera:
  motion: static       # static | moving | unknown

measurement:
  roi_polygon: null    # null = full frame
  ignore_regions: []
  regions:
    - id: region_1
      polygon: [[x, y], ...]
  counting_lines:
    - id: line_1
      points: [[x1, y1], [x2, y2]]
      direction_labels:
        side_a_to_b: direction_a
        side_b_to_a: direction_b

provenance:
  author: manual
  created_at: ...
  source_sha256: ...
```

Nếu không có manifest:

- detection/tracking/frame count vẫn chạy toàn khung;
- không sinh line count;
- không so sánh vùng;
- không đưa ra nhận xét luật hoặc hướng địa lý;
- báo cáo ghi rõ giới hạn.

---

## 9. Work packages và acceptance gates

Không dùng tiến độ theo tuần/tháng trong tài liệu này. Mỗi work package chỉ kết thúc khi acceptance gate tương ứng đạt.

### WP0 — Khóa baseline và sửa blocker hiện tại

#### Công việc

- Giữ nguyên tracker quality/fast profiles; không chạy sweep mới.
- Sửa VLM clip-grounding: model chỉ được phép claim motion nếu thực sự nhận clip frames.
- Chuyển reasoning CLI từ prompt v1 sang prompt v3 đã khóa.
- Thêm test bảo đảm CLI thực sự load prompt được khai báo.
- Tách `perception_status` khỏi `traffic_state`.
- Không cho detection silence tự động trở thành `NORMAL`.
- Làm reasoning test độc lập khỏi ignored `output/pipeline/run16` hoặc đóng gói fixture tối thiểu.
- Bổ sung hash model, config, tracker, manifest, Git commit và environment vào run provenance.
- Sửa tài liệu còn gọi tracker mặc định là ByteTrack hoặc trộn schema v2/v3.

#### Gate G0 — Baseline tái lập

- Toàn bộ unit test pass trên workspace và trên checkout không có ignored outputs.
- Có một lệnh chuẩn để chạy pipeline video.
- Config, prompt và model được ghi trong `run.json` bằng path + hash.
- VLM không thể pass motion claim nếu input thực tế chỉ là keyframe.
- Detection silence có output `UNKNOWN`/`detection_silence`, không phải `NORMAL`.

### WP1 — Hoàn thiện pipeline video UAV

#### Công việc

- Chốt measurement manifest schema, loader và validator.
- Tạo manifest cho hai video demo, không mở rộng cho toàn bộ video nguồn.
- Thêm multi-view VLM evidence:
  - full frame;
  - ROI crop;
  - tối đa một số crop sự kiện được khóa trong config;
  - fallback ROI khi detector không có bbox.
- Mỗi crop có tọa độ, source frame và evidence reference.
- Giữ numeric facts do application lắp từ deterministic event.
- Visual observations phải có confidence, evidence refs và limitation.
- Khóa output schema và CLI demo.

#### Gate G1 — Pipeline video hoàn chỉnh

- Chạy end-to-end trên ít nhất hai video UAV khác đặc tính.
- Sinh đủ video, tracks, analytics, events, evidence, summary, run record và report.
- Không crash khi manifest thiếu hoặc chỉ có full-frame mode.
- Không sinh claim theo vùng khi region không được khai báo.
- Mọi numeric fact trong báo cáo bằng đúng nguồn `event.*`.
- Video/evidence có thể truy ngược tới source frame.

### WP2 — SUMO foundation

#### Công việc

- Chọn và pin một phiên bản SUMO.
- Tạo một scenario ngã tư bốn hướng tổng hợp.
- Khóa:
  - road network;
  - lane connections;
  - vehicle types;
  - route/flow demand;
  - traffic-light program;
  - timestep;
  - seed;
  - simulation duration.
- Dùng nhiều demand profile: low, medium, high và oversaturated.
- Viết `SumoObservationAdapter` lấy trạng thái từng bước qua TraCI.
- Xuất ground truth theo schema chung.
- Viết fixed-time controller baseline.

#### Gate G2 — Mô phỏng sạch

- Scenario chạy hoàn tất không lỗi với config/seed cố định.
- Adapter xuất đúng số vehicle, ID, lane/region và speed từ SUMO.
- Zero-noise path không làm thay đổi ground truth ngoài transformation đã khai báo.
- Fixed-time controller chạy được ở mọi demand profile đã khóa.
- Run record lưu SUMO version, network hash, demand hash, seed và controller config.

### WP3 — Perception error injector

#### Công việc

- Định nghĩa error profile có version.
- Phân biệt hai loại tham số:
  - `measured`: có nguồn benchmark/annotation;
  - `assumed_sensitivity`: dùng để phân tích độ nhạy, không được gọi là đo từ CV.
- Hỗ trợ tối thiểu:
  - missed observations;
  - false observations;
  - class confusion;
  - confidence perturbation;
  - fragmentation;
  - ID switch/false reconnection;
  - observation delay;
  - count bias.
- Không giả vờ mô hình được image blur, scale shift hay occlusion vật lý nếu không có camera renderer/projection.
- Mọi stochastic operation dùng seeded RNG riêng.

#### Gate G3 — Injector đúng định nghĩa

- Error profile bằng 0 là phép đồng nhất.
- Cùng input/config/seed cho cùng output hash.
- Tỷ lệ miss, confusion, fragmentation và delay đầu ra khớp config trong dung sai đã khóa.
- Output phân biệt rõ simulator ground truth và noisy observation.
- Có unit test cho từng error operator và integration test cho chuỗi operator.

### WP4 — Perception-aware control experiment

#### Controllers

1. `fixed_time`: không dùng observation để thay đổi phase.
2. `oracle_adaptive`: dùng SUMO clean state.
3. `noisy_adaptive`: dùng observation đã chèn lỗi.
4. `reliability_aware`: dùng noisy observation; khi status không đáng tin thì fallback về fixed-time safe plan.

Không dùng reinforcement learning trong WP4. Controller thích nghi đầu tiên phải đơn giản, có thể đọc và kiểm thử bằng rule/threshold.

#### Chỉ số

- average waiting time;
- average/maximum queue length;
- throughput;
- average travel time;
- số lần đổi phase;
- fallback rate;
- thời gian ở trạng thái degraded/unknown;
- chênh lệch so với oracle và fixed-time.

#### Protocol

- Khóa scenario, demand profiles, seed list, controller và error profiles trước khi chạy bảng chính.
- Dùng cùng scenario/demand/seed cho mọi controller.
- Báo cáo phân phối qua seed, không chọn riêng run tốt nhất.
- Không chỉnh threshold sau khi xem test result; thay đổi cần mở decision record mới.

#### Gate G4 — Đánh giá mô phỏng hợp lệ

- Chạy đủ bốn controller trên toàn bộ ma trận đã khóa.
- Không có run bị âm thầm loại khỏi bảng.
- Kết quả bao gồm cả mean và độ biến thiên.
- Báo cáo rõ tham số measured và assumed.
- Claim chỉ có hiệu lực trong topology, demand, error profile và seed đã thử.

### WP5 — Handoff, demo và báo cáo

#### Deliverables

- Một video demo pipeline UAV hoàn chỉnh và một video/record mô phỏng SUMO.
- Lệnh chạy chuẩn cho video và mô phỏng.
- Config mẫu có hash.
- Output mẫu cho real-video và simulation.
- Bảng detection/tracking/counting/runtime đã có.
- Bảng oracle/noisy/reliability-aware.
- Sơ đồ kiến trúc hai nhánh.
- Báo cáo nêu kết quả dương, âm, limitation và claim boundary.

#### Gate G5 — Có thể bàn giao

- Một người khác có thể chạy theo README mà không cần biết lịch sử các session.
- Có bảng traceability: mục tiêu → module → acceptance gate → artifact.
- Không cần ignored local artifact để unit test pass.
- Không còn tài liệu chính nào mô tả tracker, prompt hoặc schema trái với code.

---

## 10. Dữ liệu cần thiết

### 10.1. Nhánh video thật

- video UAV nguồn;
- detector checkpoint đã chọn;
- tracker config đã khóa;
- measurement manifest tùy chọn;
- GT/annotation benchmark hiện có cho detection, tracking và counting;
- output run records và evidence.

### 10.2. Nhánh SUMO

SUMO cần tối thiểu:

- network (`.net.xml` hoặc source network files);
- route/trip/flow demand (`.rou.xml`);
- vehicle types;
- simulation config (`.sumocfg`);
- traffic-light program nếu scenario có đèn;
- additional detector/output definitions nếu cần;
- seed và timestep.

SUMO chính thức mô tả network và routes là input bắt buộc; traffic lights, detector và outputs là phần bổ sung. Tham khảo:

- [SUMO documentation](https://sumo.dlr.de/docs/sumo.html)
- [SUMO routes and vehicle types](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html)
- [TraCI Python interface](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html)
- [Traffic-light control tutorial](https://sumo.dlr.de/docs/Tutorials/TraCI4Traffic_Lights.html)

### 10.3. Dữ liệu dùng để hiệu chỉnh error model

Ưu tiên dùng dữ liệu đã có:

- detection recall/precision theo class và kích thước;
- confusion matrix nếu đủ dữ liệu;
- HOTA/IDF1/fragmentation/ID switch từ tracking benchmark;
- frame-count MAE/WAPE;
- line-crossing WAPE;
- latency/runtime;
- các failure case đã xác nhận trực quan.

Không suy ra `recall_by_density`, `false_reconnection_rate` hoặc occlusion probability nếu chưa có annotation tương ứng. Những tham số đó phải được gắn nhãn `assumed_sensitivity` cho đến khi có phép đo.

---

## 11. Chính sách chống experiment drift

### 11.1. Ba loại hoạt động

#### Integration/acceptance test

Bắt buộc để chứng minh module đúng interface và hệ thống chạy. Đây không phải pilot nghiên cứu.

#### Ablation có câu hỏi

Chỉ chạy khi cần trả lời một research question đã ghi trong tài liệu và có baseline/gate cố định.

#### Exploratory pilot

Mặc định ngoài scope. Chỉ được mở khi một acceptance gate thất bại và chưa có chẩn đoán rẻ hơn.

### 11.2. Điều kiện bắt buộc trước một thử nghiệm mới

Một thử nghiệm chỉ được chạy khi có đủ:

1. Gate hoặc câu hỏi nào đang bị chặn?
2. Failure đã được tái hiện bằng artifact nào?
3. Giả thuyết nguyên nhân cụ thể là gì?
4. Baseline cố định là gì?
5. Chỉ thay đổi một biến chính nào?
6. Metric chính và gate quyết định là gì?
7. Dataset/split/seed nào được phép dùng?
8. Giới hạn số lần chạy hoặc compute là gì?
9. Kết quả nào dẫn đến giữ, bỏ hoặc hoãn?
10. Run record sẽ được lưu ở đâu?

Thiếu một trong các mục trên thì không chạy.

### 11.3. Quy tắc đóng băng

- Module đạt acceptance gate thì đóng băng.
- Chỉ mở lại khi có regression tái hiện được hoặc yêu cầu phạm vi mới từ người hướng dẫn.
- Không thử vì “có thể tốt hơn”.
- Không dùng locked test để chọn kiến trúc hoặc threshold.
- Không đổi metric sau khi thấy kết quả.
- Không dùng proxy thay cho ground truth nếu proxy đã được chứng minh lệch với quan sát thật.
- Kết quả âm được ghi lại một lần rồi dừng; không tạo chuỗi biến thể không có giả thuyết mới.

### 11.4. Các nhánh đã đóng

Không mở lại trong active roadmap:

- NWD loss;
- P2 head;
- SAHI bắt buộc trong runtime chính;
- GMC như một fix mặc định;
- stillness scalar làm congestion trigger;
- TVLR Stage C;
- ReID/tracker threshold sweep;
- detector fine-tuning mới.

Nếu một nhánh phải mở lại, decision record phải chỉ rõ regression mới khác gì bằng chứng cũ.

---

## 12. Mẫu experiment/decision record

```yaml
decision_id: DEC-YYYY-NNN
title: ...
status: proposed | running | accepted | rejected | deferred

blocked_gate: G0 | G1 | G2 | G3 | G4 | G5
research_question: RQ1 | RQ2 | RQ3 | RQ4 | null
failure_artifact: ...

hypothesis: ...
baseline:
  config: ...
  artifact_hash: ...

change:
  single_primary_variable: ...

protocol:
  dataset_or_scenario: ...
  split: ...
  seeds: [...]
  primary_metric: ...
  acceptance_gate: ...
  compute_limit: ...

outcomes:
  accept_if: ...
  reject_if: ...
  stop_if: ...

result:
  run_ids: []
  metrics: {}
  conclusion: ...
  limitations: []
```

---

## 13. Reproducibility và provenance

Mỗi run video hoặc simulation phải lưu:

- run ID và timestamp UTC;
- Git commit và dirty-worktree status;
- Python/package versions;
- CUDA/GPU nếu có;
- SUMO version nếu là simulation;
- source/config/model/tracker/manifest hashes;
- scenario network/demand/controller hashes;
- seed;
- schema versions;
- command hoặc entrypoint;
- status `completed | failed | interrupted`;
- output manifest.

Không dùng path đơn lẻ như bằng chứng vì file tại path đó có thể bị thay thế.

---

## 14. Cấu trúc thư mục mục tiêu

```text
configs/
  pipeline/
  reasoning/
  simulation/
    sumo_scenario_v1.yaml
    perception_error_v1.yaml
    controllers/

simulation/
  scenarios/
    sumo/
      intersection_v1/
        network.net.xml
        routes.rou.xml
        scenario.sumocfg

src/vn_traffic/
  observation/
    schema.py
    video_adapter.py
  simulation/
    sumo_adapter.py
    error_injector.py
    controllers.py
    evaluation.py

tests/
  fixtures/
  test_observation_schema.py
  test_sumo_adapter.py
  test_error_injector.py
  test_simulation_evaluation.py

experiments/
  <run_id>/run.json

reports/
  ke-hoach-pipeline-va-mo-phong.md
```

Đây là cấu trúc mục tiêu; không tạo toàn bộ folder trước khi work package tương ứng bắt đầu.

---

## 15. Rủi ro và phương án giảm thiểu

| Rủi ro | Hệ quả | Giảm thiểu |
|---|---|---|
| Detector mất recall ở cảnh đông | Báo `NORMAL` sai | `perception_status`, `UNKNOWN`, không suy diễn đường vắng |
| Tracker identity sai dưới che khuất | Đếm lặp hoặc nối nhầm | Khóa limitation, quality/fast profiles, không tuyên bố physical identity |
| VLM chỉ thấy keyframe nhưng claim motion | Grounding sai | Bắt buộc clip frames thực sự vào model hoặc chặn motion claim |
| Không biết luật địa phương | Claim vi phạm sai | Measurement manifest không chứa luật; abstain |
| SUMO không render ảnh UAV thật | Không đo được detector AP | Nối tại tầng observation; chỉ đánh giá downstream impact |
| Error injector không phản ánh occlusion vật lý | Claim vượt bằng chứng | Phân biệt measured và assumed sensitivity |
| Scenario tổng hợp bị hiểu là địa điểm thật | Kết luận sai phạm vi | Gắn nhãn synthetic scenario, không gọi digital twin |
| Tuning controller trên test matrix | Overfit mô phỏng | Khóa development/test scenarios và seed trước |
| Tiếp tục mở pilot mới | Scope drift | Gate, decision record, freeze policy |
| Model/dataset cục bộ quá lớn | Khó bàn giao | Manifest tải lại, hash, giữ sample nhỏ; không commit weights/video lớn |

---

## 16. Claim boundary

Dự án có thể tuyên bố:

- pipeline video UAV end-to-end chạy được;
- số liệu báo cáo được lấy từ deterministic analytics và có provenance;
- hệ thống biết từ chối một số claim khi evidence không đủ;
- tracking cải thiện độ liên tục so với baseline trong benchmark đã thực hiện, kèm limitation;
- SUMO cho phép đo downstream impact của một error profile đã khai báo;
- reliability-aware fallback được đánh giá trong scenario và protocol đã khóa.

Dự án không được tuyên bố:

- có detector mới state of the art;
- tổng quát tới mọi video UAV hoặc mọi thành phố;
- biết luật giao thông của địa điểm quay;
- SUMO là digital twin của video thật;
- error injector tái tạo đầy đủ blur, scale shift và severe occlusion;
- controller tối ưu ngoài topology/demand/error profiles đã thử;
- VLM observation là ground truth.

---

## 17. Thứ tự triển khai bắt buộc

```text
WP0 — Sửa blocker và khóa baseline
  ↓ G0
WP1 — Pipeline video hoàn chỉnh
  ↓ G1
WP2 — SUMO foundation + clean adapter
  ↓ G2
WP3 — Error injector
  ↓ G3
WP4 — Oracle/noisy/reliability-aware control
  ↓ G4
WP5 — Demo, báo cáo và handoff
  ↓ G5
Hoàn thành active roadmap
```

Không bắt đầu WP2 trước khi schema chung được khóa ở WP1. Không bắt đầu controller adaptive trước khi zero-noise SUMO adapter vượt G2. Không chạy bảng chính trước khi error injector vượt G3.

---

## 18. Danh sách khởi đầu (lịch sử)

Các mục 1-7 dưới đây là backlog tại thời điểm khóa kế hoạch; chúng đã được triển khai trong nhánh video hiện tại và không còn là “việc cần làm ngay”. Nhánh SUMO trong các WP phía trên vẫn chưa được triển khai.

1. Sửa clip-grounding giữa evidence request, VLM input thực tế và validator.
2. Chuyển reasoning CLI sang prompt v3 và thêm integration test.
3. Thiết kế `perception_status` + traffic state `UNKNOWN` cho detection silence.
4. Tạo fixture reasoning tối thiểu để test không phụ thuộc `output/pipeline/run16`.
5. Bổ sung provenance hashes vào `run.json`.
6. Chốt `TrafficObservation` schema v1.
7. Sau khi G0 đạt, triển khai measurement manifest và multi-view evidence.

Không chạy thêm detector training hoặc tracker sweep trong lúc thực hiện danh sách này.

