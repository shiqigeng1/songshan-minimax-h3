"""Extract the unchanged official first-frame PNG embedded in the report."""
import argparse
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "reports/【嵩山杯】Minimax初赛阶段报告-究竟对不队.docx")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
expected = "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54"
with ZipFile(args.report) as report:
    for name in report.namelist():
        if name.startswith("word/media/"):
            data = report.read(name)
            if sha256(data).hexdigest() == expected:
                break
    else:
        raise SystemExit("Official first-frame PNG not found in this report")
if args.output.exists() and sha256(args.output.read_bytes()).hexdigest() != expected:
    raise SystemExit(f"Refusing to overwrite a different file: {args.output}")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_bytes(data)
print(args.output.resolve())
print("sha256:", expected)
