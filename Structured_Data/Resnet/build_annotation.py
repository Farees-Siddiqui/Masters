"""Hand annotation of samples/resnet.pdf, and the generator that renders it.

Every record below was read off the rendered PDF pages. The pipeline's OCR
(`samples/resnet.mistral.md`) is deliberately not used as a source, so where the
OCR is wrong this annotation still agrees with the paper.

Conventions are documented in ANNOTATION.md. In short: Title-Case keys, values
as printed, the unit in the key, and one record per distinct (key, value) pair
for the whole paper.

    python Resnet/build_annotation.py        # writes Resnet/sec_*.tex
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

# (key, value, page-first-seen). Order within a group is the paper's order.
A: list[tuple[str, str, int]] = []


def add(key: str, page: int, *values: str) -> None:
    A.extend((key, v, page) for v in values)


# --------------------------------------------------------------- bibliographic
add("Title", 1, "Deep Residual Learning for Image Recognition")
add("Author", 1, "Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun")
add("Organization", 1, "Microsoft Research")
add("Email", 1, "{kahe, v-xiangz, v-shren, jiansun}@microsoft.com")
add("Identifier", 1, "arXiv:1512.03385v1")
add("Category", 1, "cs.CV")
add("Date", 1, "10 Dec 2015")
add("Date", 11, "2015-11-26")
add("URL", 1, "http://image-net.org/challenges/LSVRC/2015/",
    "http://mscoco.org/dataset/#detections-challenge2015")
add("URL", 11, "http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4",
    "http://host.robots.ox.ac.uk:8080/anonymous/3OJ4OJ.html")
add("Software", 2, "Caffe")

# -------------------------------------------------------------------- datasets
add("Dataset", 1, "ImageNet", "CIFAR-10", "COCO")
add("Dataset", 4, "ImageNet 2012")
add("Dataset", 7, "MNIST")
add("Dataset", 8, "PASCAL VOC 2007", "PASCAL VOC 2012", "MS COCO")
add("Dataset", 11, "ImageNet DET")
add("Task", 1, "Classification", "Object detection", "ImageNet detection",
    "ImageNet localization", "COCO detection", "COCO segmentation",
    "image classification", "visual recognition")
add("Task", 2, "image retrieval", "vector quantization")
add("Task", 12, "localization")
add("Split", 7, "45k/5k train/val split")
add("Split", 8, "VOC 07 test", "VOC 12 test")
add("Split", 10, "07+12", "07++12")
add("Split", 11, "COCO train", "COCO val", "COCO trainval", "COCO test-dev",
    "COCO+07+12", "COCO+07++12", "val1/val2")
add("Size", 4, "1.28 million training images", "50k validation images",
    "100k test images")
add("Size", 7, "50k training images", "10k test images")
add("Size", 10, "5k trainval", "16k trainval", "10k trainval+test",
    "80k images", "40k images")
add("Size", 11, "80k+40k trainval", "20k test-dev")
add("Classes", 4, "1000 classes")
add("Classes", 7, "10 classes")
add("Classes", 10, "80 object categories")
add("Classes", 11, "200 object categories")
add("Classes", 12, "1000-class")

# ---------------------------------------------------------------- architecture
add("Architecture", 1, "deep convolutional neural networks", "VGG nets",
    "residual nets", "plain")
add("Architecture", 2, "VLAD", "Fisher Vector", "Multigrid", "MLPs",
    "inception", "highway networks", "ResNet")
add("Architecture", 3, "VGG-19")
add("Architecture", 4, "34-layer plain", "34-layer residual")
add("Architecture", 5, "plain-18", "plain-34", "ResNet-18", "ResNet-34")
add("Architecture", 6, "VGG-16", "GoogLeNet", "PReLU-net", "ResNet-34 A",
    "ResNet-34 B", "ResNet-34 C", "ResNet-50", "ResNet-101", "ResNet-152",
    "BN-inception", "ResNet-50/101/152", "bottleneck")
add("Architecture", 7, "VGG-16/19", "Maxout", "NIN", "DSN", "FitNet", "Highway",
    "ResNet-20", "ResNet-56", "ResNet-110", "1202-layer network")
add("Architecture", 8, "Faster R-CNN")
add("Architecture", 10, "Fast R-CNN", "ResNet-50/101")
add("Architecture", 12, "OverFeat", "R-CNN", "RPN")
add("Component", 2, "shortcut connections", "identity shortcuts", "weight layer")
add("Component", 3, "ReLU", "global average pooling",
    "1000-way fully-connected layer", "softmax")
add("Component", 4, "batch normalization (BN)", "projection shortcut",
    "zero-padding shortcuts")
add("Component", 5, "max pool")
add("Component", 7, "10-way fully-connected layer")
add("Component", 10, "region proposal network (RPN)", "RoI pooling",
    "Spatial Pyramid Pooling")
add("Component", 12, "anchor", "cls", "reg")
add("Depth", 1, "152", "100", "1000", "20", "56", "sixteen", "thirty")
add("Depth", 3, "34")
add("Depth", 5, "18")
add("Depth", 6, "50", "101")
add("Depth", 7, "19", "32", "44", "110", "1202")
add("Layers", 3, "34", "two or three layers")
add("Layers", 6, "3 layers")
add("Layers", 7, "6n+2", "3n", "1+2n", "2n")
add("Layers", 10, "91 conv layers", "13 conv layers")
add("Filters", 3, "3x3")
add("Filters", 4, "1x1")
add("Filters", 5, "7x7", "64", "128", "256", "512", "1024", "2048")
add("Filters", 7, "{16, 32, 64}")
add("Output size", 5, "112x112", "56x56", "28x28", "14x14", "7x7", "1x1")
add("Output size", 7, "32x32", "16x16", "8x8")
add("FLOPs", 3, "3.6 billion", "19.6 billion")
add("FLOPs", 5, "1.8x10^9", "3.6x10^9", "3.8x10^9", "7.6x10^9", "11.3x10^9")
add("FLOPs", 7, "3.8 billion", "11.3 billion", "15.3/19.6 billion")
add("Parameters", 7, "2.5M", "2.3M", "1.25M", "0.27M", "0.46M", "0.66M",
    "0.85M", "1.7M", "19.4M")
add("Stride", 3, "2")
add("Stride", 10, "16 pixels")

# ---------------------------------------------------------------------- method
add("Method", 1, "residual learning framework", "normalized initialization",
    "intermediate normalization layers", "stochastic gradient descent (SGD)",
    "backpropagation")
add("Method", 2, "deep residual learning", "hierarchical basis preconditioning")
add("Method", 4, "10-crop testing")
add("Method", 7, "data augmentation")
add("Method", 8, "maxout", "dropout")
add("Method", 10, "box refinement", "global context", "multi-scale testing",
    "non-maximum suppression (NMS)", "box voting")
add("Method", 11, "ensemble")
add("Method", 12, "per-class regression (PCR)", "oracle", "1-crop", "dense")
add("Problem", 1, "vanishing/exploding gradients", "degradation")
add("Problem", 7, "optimization difficulty")
add("Problem", 8, "overfitting")
add("Concept", 1, "depth", "low/mid/high-level features", "identity mapping")
add("Concept", 3, "residual function", "residual mapping", "underlying mapping")
add("Concept", 6, "option A", "option B", "option C", "bottleneck design")
add("Equation", 3, "(1)", "(2)")
add("Figure", 1, "Fig. 1", "Fig. 4")
add("Figure", 2, "Fig. 2")
add("Figure", 3, "Fig. 3", "Fig. 5", "Fig. 7")
add("Figure", 7, "Fig. 6")
add("Table", 4, "Table 1", "Table 2")
add("Table", 6, "Table 3", "Table 4", "Table 5")
add("Table", 7, "Table 6")
add("Table", 8, "Table 7", "Table 8")
add("Table", 10, "Table 9")
add("Table", 11, "Table 10", "Table 11", "Table 12")
add("Table", 12, "Table 13", "Table 14")

# -------------------------------------------------------------------- training
add("Learning rate", 4, "0.1")
add("Learning rate", 7, "0.01")
add("Learning rate", 10, "0.001", "0.0001")
add("Weight decay", 4, "0.0001")
add("Momentum", 4, "0.9")
add("Batch size", 4, "256")
add("Batch size", 7, "128")
add("Batch size", 10, "8 images", "16 images")
add("Iterations", 4, "60 x 10^4")
add("Iterations", 7, "32k", "48k", "64k", "400 iterations")
add("Iterations", 10, "240k", "80k")
add("Crop", 4, "224x224")
add("Crop", 7, "32x32")
add("Scale", 4, "[256, 480]", "{224, 256, 384, 480, 640}")
add("Scale", 10, "600 pixels", "{200, 400, 600, 800, 1000}")
add("Augmentation", 4, "horizontal flip", "per-pixel mean subtracted",
    "color augmentation")
add("Augmentation", 7, "4 pixels are padded")
add("Hardware", 7, "two GPUs")
add("Hardware", 10, "8-GPU implementation", "1 per GPU")
add("Threshold", 10, "0.3")
add("Ratio", 12, "1:1")
add("Proposals", 10, "300 proposals")
add("Proposals", 12, "200 proposals")
add("Anchors", 12, "8 anchors")

# --------------------------------------------------------------------- results
add("Metric", 4, "top-1 error", "top-5 error")
add("Metric", 5, "training error", "test error", "validation error")
add("Metric", 8, "mAP@.5", "mAP@[.5, .95]")
add("Metric", 12, "top-5 localization err", "LOC error on GT CLS")
add("Error", 1, "3.57%")
# Table 2 (p.5), Table 3 (p.6), Table 4 (p.6), Table 5 (p.6)
add("Error", 5, "27.94", "27.88", "28.54", "25.03")
add("Error", 6, "28.07", "9.33", "9.15", "24.27", "7.38", "10.02", "7.76",
    "24.52", "7.46", "24.19", "7.40", "22.85", "6.71", "21.75", "6.05",
    "21.43", "5.71", "8.43", "7.89", "24.4", "7.1", "21.59", "21.99", "5.81",
    "21.84", "21.53", "5.60", "20.74", "5.25", "19.87", "4.60", "19.38",
    "4.49", "7.32", "6.66", "6.8", "4.94", "4.82", "3.57")
# Table 6 (p.7)
add("Error", 7, "9.38", "8.81", "8.22", "8.39", "7.54 (7.72±0.16)", "8.80",
    "8.75", "7.51", "7.17", "6.97", "6.43 (6.61±0.16)", "7.93")
add("Error", 7, "4.49%")
add("Error", 8, "6.43%", "7.93%", "60%", "0.1%")
add("Error", 12, "33.1%", "13.3%", "11.7%", "14.4%", "10.6%", "9.0%", "4.6%",
    "33.1", "13.3", "11.7", "14.4", "10.6", "8.9", "30.0", "29.9", "26.7",
    "26.9", "25.3", "9.0")
add("Error", 7, "80%")
add("Error", 7, "90%")
# Tables 7-9, 12 (pp.8, 11)
add("mAP", 8, "73.2", "70.4", "76.4", "73.8", "41.5", "21.2", "48.4", "27.2")
add("mAP", 11, "49.9", "29.9", "51.1", "30.0", "53.3", "32.2", "53.8", "32.5",
    "55.7", "34.9", "59.0", "37.4", "43.9", "60.5", "58.8", "63.6", "62.1")
add("mAP", 11, "85.6", "83.8")
add("Improvement", 1, "28%")
add("Improvement", 3, "18%")
add("Improvement", 5, "2.8%")
add("Improvement", 6, "3.5%")
add("Improvement", 8, "6.0%")
add("Improvement", 10, "6%", "6.9%", ">3%", "about 2 points", "about 1 point")
add("Improvement", 11, "over 2 points", "10 points")
add("Improvement", 12, "8.5 points", "64%")
add("Comparison", 1, "8x")
add("Competition", 1, "ILSVRC 2015", "ILSVRC & COCO 2015")
add("Competition", 6, "ILSVRC'14", "ILSVRC'15")
add("Competition", 11, "COCO 2015")
add("Competition", 12, "ILSVRC'13", "ILSVRC 14")
add("Placement", 1, "1st place", "1st places")
add("Placement", 12, "second place")

# Tables 10 and 11 (p.11): per-class AP. Twenty classes, three systems each.
_CLS = ("aero bike bird boat bottle bus car cat chair cow table dog horse "
        "mbike person plant sheep sofa train tv").split()
_T10 = ["76.5 79.0 70.9 65.5 52.1 83.1 84.7 86.4 52.0 81.9 65.7 84.8 84.6 77.5 76.7 38.8 73.6 73.9 83.0 72.6",
        "79.8 80.7 76.2 68.3 55.9 85.1 85.3 89.8 56.7 87.8 69.4 88.3 88.9 80.9 78.4 41.7 78.6 79.8 85.3 72.0",
        "90.0 89.6 87.8 80.8 76.1 89.9 89.9 89.6 75.5 90.0 80.7 89.6 90.3 89.1 88.7 65.4 88.1 85.6 89.0 86.8"]
_T11 = ["84.9 79.8 74.3 53.9 49.8 77.5 75.9 88.5 45.6 77.1 55.3 86.9 81.7 80.9 79.6 40.1 72.6 60.9 81.2 61.5",
        "86.5 81.6 77.2 58.0 51.0 78.6 76.6 93.2 48.6 80.4 59.0 92.1 85.3 84.8 80.7 48.1 77.3 66.5 84.7 65.6",
        "92.1 88.4 84.8 75.9 71.4 86.3 87.8 94.2 66.8 89.4 69.2 93.9 91.9 90.9 89.6 67.9 88.2 76.8 90.3 80.0"]
for row in _T10 + _T11:
    add("AP", 11, *row.split())
add("Class", 11, *_CLS)

# ------------------------------------------------------------------ rendering

SECTIONS: list[tuple[str, str, tuple[str, ...]]] = [
    ("sec_bibliographic", "Bibliographic",
     ("Title", "Author", "Organization", "Email", "Identifier", "Category",
      "Date", "URL", "Software")),
    ("sec_datasets", "Datasets and tasks",
     ("Dataset", "Task", "Split", "Size", "Classes")),
    ("sec_architecture", "Architecture",
     ("Architecture", "Component", "Depth", "Layers", "Filters", "Output size",
      "FLOPs", "Parameters", "Stride")),
    ("sec_method", "Method",
     ("Method", "Problem", "Concept", "Equation", "Figure", "Table")),
    ("sec_training", "Training setup",
     ("Learning rate", "Weight decay", "Momentum", "Batch size", "Iterations",
      "Crop", "Scale", "Augmentation", "Hardware", "Threshold", "Ratio",
      "Proposals", "Anchors")),
    ("sec_results", "Results",
     ("Metric", "Error", "mAP", "AP", "Class", "Improvement", "Comparison",
      "Competition", "Placement")),
]

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_UNI = {"±": r"$\pm$", "×": r"$\times$"}


def tex(s: str) -> str:
    out = []
    for ch in s:
        out.append(_UNI.get(ch) or _ESC.get(ch) or ch)
    return "".join(out)


def main() -> int:
    seen: set[tuple[str, str]] = set()
    records: list[tuple[str, str, int]] = []
    for key, value, page in A:                      # one record per distinct pair
        if (key, value) not in seen:
            seen.add((key, value))
            records.append((key, value, page))

    total = 0
    for stem, heading, keys in SECTIONS:
        lines = [f"\\section{{{heading}}}", ""]
        for key in keys:
            rows = [r for r in records if r[0] == key]
            if not rows:
                continue
            lines += [f"\\begin{{kvtable}}{{{tex(key)}}}{{tab:{stem}-{tex(key).replace(' ', '-').lower()}}}"]
            for _, value, page in rows:
                # URLs are long and unbreakable once escaped; \url sets them
                # verbatim and lets them wrap inside the column.
                cell = f"\\url{{{value}}}" if key == "URL" else tex(value)
                lines.append(f"\\K{{{tex(key)}}} & {cell} & \\Sd{{p.{page}}}\\\\")
            lines += ["\\end{kvtable}", ""]
            total += len(rows)
        (OUT / f"{stem}.tex").write_text("\n".join(lines), encoding="utf-8")
        n = sum(1 for r in records if r[0] in keys)
        print(f"  {stem:22s} {n:4d}")

    print(f"\n{len(records)} distinct records "
          f"({len(A) - len(records)} duplicates collapsed), "
          f"{len({r[0] for r in records})} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
