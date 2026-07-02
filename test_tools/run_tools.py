import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# Dynamically calculate project paths relative to this script's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")


def clean_and_parse_line(line):
    """Fixes image path without harming internal metadata"""
    match = re.search(r'"image_path"\s*:\s*"([^"]+)"', line)
    if match:
        raw_path = match.group(1)
        fixed_path = raw_path.replace("\\", "\\\\")
        line = line.replace(raw_path, fixed_path)

    data = json.loads(line)
    return data


def parse_config(config_path):
    cases = []
    current_category = "UNKNOWN"

    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        return cases

    with open(config_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                comment_content = line.replace("#", "").strip()
                if ":" in comment_content:
                    current_category = comment_content.split(":")[0].strip()
                continue

            if "image_path" not in line:
                continue

            try:
                data = clean_and_parse_line(line)

                image_path = data.get("image_path", "")
                # Resolve relative image addressing based on the project root
                if image_path and not os.path.isabs(image_path):
                    image_path = os.path.abspath(
                        os.path.join(ROOT_DIR, image_path)
                    )

                image_path = os.path.normpath(image_path)

                if image_path and not image_path.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    image_path += ".jpg"

                metadata_raw = data.get("metadata", "{}")
                if isinstance(metadata_raw, str):
                    metadata_json = json.loads(metadata_raw)
                else:
                    metadata_json = metadata_raw

                cases.append(
                    {
                        "category": current_category,
                        "image_path": image_path,
                        "metadata_str": json.dumps(metadata_json),
                        "prompt": data.get(
                            "prompt", "Is this a skin cancer?"
                        ),
                    }
                )
            except Exception as e:
                if "test.jpg" not in line:
                    print(
                        f"⚠️ Line {line_num} skipped due to parsing issue: {e}"
                    )
                continue
    return cases


def run_script(script_name, image_path, metadata_str):
    script_path = os.path.join(TOOLS_DIR, script_name)
    if not os.path.exists(script_path):
        return f"FAILED ({script_name} not found)", 0.0

    cmd = [
        sys.executable,
        script_path,
        "--image",
        image_path,
        "--metadata",
        metadata_str,
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=150,  # Longer timeout to handle heavy processing of the High layer
        )
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "SUCCESS (No stdout)", round(
                elapsed_time, 2
            )
        else:
            return (
                f"ERROR (Exit Code {result.returncode}): {result.stderr.strip()}",
                round(elapsed_time, 2),
            )

    except subprocess.TimeoutExpired:
        return "TIMEOUT ERROR (Exceeded 150s)", round(
            time.time() - start_time, 2
        )
    except Exception as e:
        return f"EXECUTION FAILED: {str(e)}", round(time.time() - start_time, 2)


def main():
    config_file = os.path.join(ROOT_DIR, "prompt.txt")
    script_fast = "skin_lesion_fast.py"
    script_mid = "skin_lesion_mid.py"
    script_high = "skin_lesion_high.py"

    print("📌 Loading configuration from prompt.txt...")
    test_cases = parse_config(config_file)
    print(f"✅ Found {len(test_cases)} valid test cases.\n")

    if not test_cases:
        print("❌ No test cases to run. Please check prompt.txt format.")
        return

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save report files directly within the test_tools folder
    report_txt_path = os.path.join(CURRENT_DIR, f"test_report_{timestamp}.txt")
    report_csv_path = os.path.join(CURRENT_DIR, f"test_results_{timestamp}.csv")

    with open(report_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"=== SKIN LESION CLASSIFIER BENCHMARK REPORT ===\n")
        txt_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txt_file.write(f"Total Test Cases: {len(test_cases)}\n")
        txt_file.write(f"=" * 50 + "\n\n")

        for idx, case in enumerate(test_cases, 1):
            img_name = os.path.basename(case["image_path"])
            print(
                f"🔄 [{idx}/{len(test_cases)}] Processing {img_name} ({case['category']})..."
            )

            print(f"   -> Running {script_fast}...")
            fast_out, fast_time = run_script(
                script_fast, case["image_path"], case["metadata_str"]
            )

            print(f"   -> Running {script_mid}...")
            mid_out, mid_time = run_script(
                script_mid, case["image_path"], case["metadata_str"]
            )

            print(f"   -> Running {script_high}...")
            high_out, high_time = run_script(
                script_high, case["image_path"], case["metadata_str"]
            )

            record = {
                "ID": idx,
                "Category": case["category"],
                "Image": img_name,
                "Fast_Time(s)": fast_time,
                "Mid_Time(s)": mid_time,
                "High_Time(s)": high_time,
                "Fast_Result": fast_out,
                "Mid_Result": mid_out,
                "High_Result": high_out,
            }
            results.append(record)

            txt_file.write(
                f"Case #{idx} | Category: {case['category']} | Image: {img_name}\n"
            )
            txt_file.write(f"Path: {case['image_path']}\n")
            txt_file.write(f"Metadata: {case['metadata_str']}\n")
            txt_file.write(f"-" * 30 + "\n")
            txt_file.write(f"⏱️ [FAST] Time: {fast_time}s\n📋 Output:\n{fast_out}\n\n")
            txt_file.write(f"⏱️ [MID]  Time: {mid_time}s\n📋 Output:\n{mid_out}\n\n")
            txt_file.write(f"⏱️ [HIGH] Time: {high_time}s\n📋 Output:\n{high_out}\n")
            txt_file.write(f"=" * 50 + "\n\n")
            txt_file.flush()

    keys = [
        "ID",
        "Category",
        "Image",
        "Fast_Time(s)",
        "Mid_Time(s)",
        "High_Time(s)",
        "Fast_Result",
        "Mid_Result",
        "High_Result",
    ]
    with open(
        report_csv_path, "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        dict_writer = csv.DictWriter(csv_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print("\n" + "=%=" * 15)
    print(f"🚀 Testing completed successfully!")
    print(f"📄 Detailed text report saved to: {report_txt_path}")
    print(f"📊 Excel-ready CSV summary saved to: {report_csv_path}")
    print("=%=" * 15)


if __name__ == "__main__":
    main()