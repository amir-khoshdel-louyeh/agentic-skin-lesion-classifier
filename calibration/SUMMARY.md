# Local calibration report (research use only)

Generated: 2026-10-19T14:44:41.502196+00:00

Gate: n >= 150, accuracy >= 0.6, ECE <= 0.1. Tools that fail stay out of the pool.

| tool | n | accuracy | balanced acc | ECE | mean conf | gate |
|------|---|----------|--------------|-----|-----------|------|
| dermai-b0 | 175 | 0.5714 | 0.5714 | 0.1803 | 0.4368 | FAIL |
| ensemble-high | 175 | 0.7200 | 0.7200 | 0.0936 | 0.6339 | PASS |
| ham10000-cnn | 175 | 0.6514 | 0.6514 | 0.0983 | 0.7054 | PASS |
| isic-efficientnet-b4 | 175 | 0.6686 | 0.6686 | 0.0579 | 0.6688 | PASS |
| isic-resnet50 | 175 | 0.7371 | 0.7371 | 0.0895 | 0.7735 | PASS |
| isic-vit | 175 | 0.7086 | 0.7086 | 0.0942 | 0.7926 | PASS |
| multimodal-fusion | 174 | 0.6437 | 0.6424 | 0.1053 | 0.6379 | FAIL |
| skin-lesion-high | 175 | 0.6686 | 0.6686 | 0.1290 | 0.5811 | FAIL |
| skin-lesion-mid | 175 | 0.5714 | 0.5714 | 0.1803 | 0.4368 | FAIL |

## Per-class recall

- **dermai-b0** (FAIL): actinic 0.24, basal 0.52, benign 0.68, dermatofibroma 0.52, melanoma 0.36, melanocytic 0.92, vascular 0.76
- **ensemble-high** (PASS): actinic 0.68, basal 0.68, benign 0.48, dermatofibroma 0.84, melanoma 0.68, melanocytic 0.72, vascular 0.96
- **ham10000-cnn** (PASS): actinic 0.48, basal 0.64, benign 0.48, dermatofibroma 0.76, melanoma 0.76, melanocytic 0.68, vascular 0.76
- **isic-efficientnet-b4** (PASS): actinic 0.60, basal 0.92, benign 0.60, dermatofibroma 0.40, melanoma 0.52, melanocytic 0.84, vascular 0.80
- **isic-resnet50** (PASS): actinic 0.72, basal 0.84, benign 0.60, dermatofibroma 0.64, melanoma 0.56, melanocytic 0.84, vascular 0.96
- **isic-vit** (PASS): actinic 0.52, basal 0.80, benign 0.64, dermatofibroma 0.52, melanoma 0.68, melanocytic 0.84, vascular 0.96
- **multimodal-fusion** (FAIL): actinic 0.76, basal 0.48, benign 0.42, dermatofibroma 0.84, melanoma 0.44, melanocytic 0.64, vascular 0.92
- **skin-lesion-high** (FAIL): actinic 0.68, basal 0.44, benign 0.52, dermatofibroma 0.76, melanoma 0.64, melanocytic 0.80, vascular 0.84
- **skin-lesion-mid** (FAIL): actinic 0.24, basal 0.52, benign 0.68, dermatofibroma 0.52, melanoma 0.36, melanocytic 0.92, vascular 0.76

## Failures

- dermai-b0: accuracy=0.5714 < 0.6; ece=0.1803 > 0.1
- multimodal-fusion: ece=0.1053 > 0.1
- skin-lesion-high: ece=0.1290 > 0.1
- skin-lesion-mid: accuracy=0.5714 < 0.6; ece=0.1803 > 0.1

Full per-tool metrics (reliability bins, confusion matrix, per-class precision/F1) live in the JSON files next to this summary.

Research use only — not a medical diagnosis.
