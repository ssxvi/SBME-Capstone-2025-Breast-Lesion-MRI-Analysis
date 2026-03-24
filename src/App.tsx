import React, { useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  FileImage,
  FolderOpen,
  Play,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Download,
  FileText,
  ShieldAlert,
  Scissors,
  Brain,
  Activity,
  Layers,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";

const STEP_META = [
  {
    key: "validate",
    title: "Validate input",
    desc: "Accept only NIfTI or DICOM input.",
    icon: ShieldAlert,
  },
  {
    key: "convert",
    title: "DICOM → NIfTI",
    desc: "Group DICOM series and convert before downstream analysis.",
    icon: RefreshCw,
  },
  {
    key: "crop",
    title: "Breast cropping",
    desc: "Optional manual cropping when external chest structures are present.",
    icon: Scissors,
  },
  {
    key: "lesionDetection",
    title: "Lesion vs no lesion",
    desc: "EfficientNet screening model.",
    icon: Brain,
  },
  {
    key: "lesionType",
    title: "Benign vs malignant",
    desc: "Runs only when lesion probability exceeds threshold.",
    icon: Activity,
  },
  {
    key: "segmentation",
    title: "nnU-Net segmentation",
    desc: "Generate lesion mask for lesion-positive cases.",
    icon: Layers,
  },
  {
    key: "report",
    title: "Export report",
    desc: "Produce CSV/JSON summary for the user.",
    icon: FileText,
  },
];

const ACCEPTED_EXTENSIONS = [".nii", ".nii.gz", ".dcm"];

function detectInputType(files) {
  if (!files?.length) return null;

  const names = Array.from(files).map((f) => f.name.toLowerCase());
  const hasNifti = names.some((n) => n.endsWith(".nii") || n.endsWith(".nii.gz"));
  const allDicom = names.every((n) => n.endsWith(".dcm"));

  if (hasNifti) return "nifti";
  if (allDicom) return "dicom";
  return "invalid";
}

function downloadTextFile(filename, text, type = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function toCSV(rows) {
  const header = Object.keys(rows[0] || {});
  const escape = (value) => {
    const str = String(value ?? "");
    return `"${str.replaceAll('"', '""')}"`;
  };
  return [header.join(","), ...rows.map((row) => header.map((h) => escape(row[h])).join(","))].join("\n");
}

function StatusBadge({ status }) {
  if (status === "complete") return <Badge className="gap-1"><CheckCircle2 className="h-3.5 w-3.5" />Complete</Badge>;
  if (status === "running") return <Badge variant="secondary" className="gap-1"><Loader2 className="h-3.5 w-3.5 animate-spin" />Running</Badge>;
  if (status === "skipped") return <Badge variant="outline">Skipped</Badge>;
  if (status === "error") return <Badge variant="destructive" className="gap-1"><AlertCircle className="h-3.5 w-3.5" />Error</Badge>;
  return <Badge variant="outline">Pending</Badge>;
}

export default function BreastImagingPipelineUI() {
  const inputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [inputType, setInputType] = useState(null);
  const [validationMessage, setValidationMessage] = useState("No files uploaded yet.");
  const [hasExternalChest, setHasExternalChest] = useState(false);
  const [reportFormat, setReportFormat] = useState("csv");
  const [pipelineName, setPipelineName] = useState("Breast Lesion MRI/Ultrasound Pipeline");
  const [notes, setNotes] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stepStatuses, setStepStatuses] = useState(
    STEP_META.reduce((acc, step) => ({ ...acc, [step.key]: "pending" }), {})
  );
  const [outputs, setOutputs] = useState(null);

  const fileSummary = useMemo(() => {
    if (!files.length) return "No files selected";
    if (inputType === "dicom") return `${files.length} DICOM file${files.length > 1 ? "s" : ""} selected`;
    if (inputType === "nifti") return `${files.length} NIfTI file${files.length > 1 ? "s" : ""} selected`;
    return `${files.length} file${files.length > 1 ? "s" : ""} selected`;
  }, [files, inputType]);

  const handleFiles = (fileList) => {
    const nextFiles = Array.from(fileList || []);
    setFiles(nextFiles);
    setOutputs(null);
    setProgress(0);
    setStepStatuses(STEP_META.reduce((acc, step) => ({ ...acc, [step.key]: "pending" }), {}));

    const detected = detectInputType(nextFiles);
    setInputType(detected);

    if (!nextFiles.length) {
      setValidationMessage("No files uploaded yet.");
      return;
    }

    if (detected === "nifti") {
      setValidationMessage("Valid NIfTI input detected. Ready to continue.");
      return;
    }

    if (detected === "dicom") {
      setValidationMessage("Valid DICOM series detected. These will be grouped and converted to NIfTI before inference.");
      return;
    }

    setValidationMessage("Unsupported input. Please upload only NIfTI (.nii / .nii.gz) or DICOM (.dcm) files.");
  };

  const markStep = (key, status) => {
    setStepStatuses((prev) => ({ ...prev, [key]: status }));
  };

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const resetPipeline = () => {
    setOutputs(null);
    setProgress(0);
    setIsRunning(false);
    setStepStatuses(STEP_META.reduce((acc, step) => ({ ...acc, [step.key]: "pending" }), {}));
  };

  const runPipeline = async () => {
    if (!files.length) {
      setValidationMessage("Please upload a NIfTI or DICOM input first.");
      return;
    }

    if (inputType === "invalid" || !inputType) {
      markStep("validate", "error");
      setValidationMessage("Only NIfTI (.nii / .nii.gz) and DICOM (.dcm) inputs are accepted.");
      return;
    }

    setIsRunning(true);
    setOutputs(null);
    setProgress(5);
    setStepStatuses(STEP_META.reduce((acc, step) => ({ ...acc, [step.key]: "pending" }), {}));

    try {
      markStep("validate", "running");
      await wait(600);
      markStep("validate", "complete");
      setProgress(15);

      if (inputType === "dicom") {
        markStep("convert", "running");
        await wait(900);
        markStep("convert", "complete");
      } else {
        markStep("convert", "skipped");
      }
      setProgress(30);

      if (hasExternalChest) {
        markStep("crop", "running");
        await wait(900);
        markStep("crop", "complete");
      } else {
        markStep("crop", "skipped");
      }
      setProgress(45);

      markStep("lesionDetection", "running");
      await wait(900);
      const lesionProbability = 0.73;
      const lesionLabel = lesionProbability > 0.5 ? "Lesion" : "No lesion";
      markStep("lesionDetection", "complete");
      setProgress(60);

      let lesionTypeProbability = null;
      let lesionTypeLabel = "Not run";
      let segmentationStatus = "Not run";
      let maskFilename = "N/A";

      if (lesionProbability > 0.5) {
        markStep("lesionType", "running");
        await wait(850);
        lesionTypeProbability = 0.81;
        lesionTypeLabel = lesionTypeProbability > 0.5 ? "Malignant" : "Benign";
        markStep("lesionType", "complete");
        setProgress(78);

        markStep("segmentation", "running");
        await wait(1000);
        segmentationStatus = "Mask generated";
        maskFilename = "predicted_mask.nii.gz";
        markStep("segmentation", "complete");
      } else {
        markStep("lesionType", "skipped");
        markStep("segmentation", "skipped");
      }

      setProgress(92);
      markStep("report", "running");
      await wait(500);

      const reportRows = [
        {
          pipeline_name: pipelineName,
          uploaded_input_type: inputType,
          uploaded_files: files.map((f) => f.name).join(" | "),
          external_chest_present: hasExternalChest ? "yes" : "no",
          dicom_to_nifti: inputType === "dicom" ? "performed" : "not_needed",
          manual_cropping: hasExternalChest ? "breast_seg.py" : "not_needed",
          lesion_screening_label: lesionLabel,
          lesion_screening_probability: lesionProbability.toFixed(3),
          lesion_type_label: lesionTypeLabel,
          lesion_type_probability: lesionTypeProbability !== null ? lesionTypeProbability.toFixed(3) : "N/A",
          segmentation_result: segmentationStatus,
          segmentation_mask: maskFilename,
          notes,
        },
      ];

      const nextOutputs = {
        summary: {
          lesionLabel,
          lesionProbability,
          lesionTypeLabel,
          lesionTypeProbability,
          segmentationStatus,
          maskFilename,
        },
        reportRows,
      };

      setOutputs(nextOutputs);
      markStep("report", "complete");
      setProgress(100);
    } catch (error) {
      console.error(error);
      setValidationMessage("Pipeline execution failed. Replace the mock handlers with your backend calls.");
    } finally {
      setIsRunning(false);
    }
  };

  const exportReport = () => {
    if (!outputs?.reportRows?.length) return;

    if (reportFormat === "csv") {
      downloadTextFile("pipeline_report.csv", toCSV(outputs.reportRows), "text/csv;charset=utf-8");
      return;
    }

    downloadTextFile("pipeline_report.json", JSON.stringify(outputs.reportRows, null, 2), "application/json;charset=utf-8");
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="space-y-6">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <Card className="rounded-2xl border-0 shadow-sm">
              <CardHeader>
                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                  <div className="space-y-2">
                    <Badge variant="outline" className="w-fit">Frontend prototype</Badge>
                    <CardTitle className="text-2xl">Breast imaging AI pipeline UI</CardTitle>
                    <CardDescription className="max-w-3xl text-sm leading-6">
                      Upload breast MRI or ultrasound imaging inputs, validate whether they are NIfTI or DICOM,
                      trigger optional cropping, route through lesion screening and subtype classification,
                      then generate segmentation and a downloadable report.
                    </CardDescription>
                  </div>
                  <div className="rounded-2xl border bg-white px-4 py-3 text-sm shadow-sm">
                    <div className="font-medium">Accepted input types</div>
                    <div className="mt-1 text-slate-600">{ACCEPTED_EXTENSIONS.join(", ")}</div>
                  </div>
                </div>
              </CardHeader>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
            <Card className="rounded-2xl border-0 shadow-sm">
              <CardHeader>
                <CardTitle>1. Input and pipeline settings</CardTitle>
                <CardDescription>
                  This UI is wired as a frontend prototype. Replace the mocked delays and outputs with your backend endpoints when the pipeline is ready.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="pipeline-name">Pipeline name</Label>
                    <Input
                      id="pipeline-name"
                      value={pipelineName}
                      onChange={(e) => setPipelineName(e.target.value)}
                      placeholder="Enter pipeline title"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Report export format</Label>
                    <RadioGroup value={reportFormat} onValueChange={setReportFormat} className="flex gap-6 pt-2">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="csv" id="csv" />
                        <Label htmlFor="csv">CSV</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="json" id="json" />
                        <Label htmlFor="json">JSON</Label>
                      </div>
                    </RadioGroup>
                  </div>
                </div>

                <div className="space-y-3">
                  <Label>Upload imaging input</Label>
                  <div className="rounded-2xl border border-dashed bg-white p-6">
                    <input
                      ref={inputRef}
                      type="file"
                      multiple
                      className="hidden"
                      accept=".nii,.nii.gz,.dcm"
                      onChange={(e) => handleFiles(e.target.files)}
                    />
                    <div className="flex flex-col items-center justify-center gap-3 text-center">
                      <div className="rounded-full bg-slate-100 p-4">
                        <Upload className="h-6 w-6" />
                      </div>
                      <div>
                        <div className="font-medium">Upload NIfTI file or DICOM series</div>
                        <div className="text-sm text-slate-500">For DICOM, select all files from one series together.</div>
                      </div>
                      <div className="flex flex-wrap justify-center gap-3">
                        <Button onClick={() => inputRef.current?.click()}>
                          <FolderOpen className="mr-2 h-4 w-4" />
                          Choose files
                        </Button>
                        <Button variant="outline" onClick={() => handleFiles([])}>
                          Clear
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border bg-white p-4">
                    <div className="mb-2 flex items-center gap-2 font-medium">
                      <FileImage className="h-4 w-4" />
                      Uploaded file summary
                    </div>
                    <div className="text-sm text-slate-600">{fileSummary}</div>
                    <div className="mt-2 text-sm text-slate-600">Detected input type: <span className="font-medium uppercase">{inputType || "N/A"}</span></div>
                  </div>
                  <div className="rounded-2xl border bg-white p-4">
                    <div className="mb-2 font-medium">Validation status</div>
                    <div className="text-sm text-slate-600">{validationMessage}</div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-2xl border bg-white p-4">
                  <div className="space-y-1">
                    <Label htmlFor="external-parts" className="text-sm font-medium">
                      Does the image include external chest parts?
                    </Label>
                    <p className="text-sm text-slate-500">
                      If enabled, the UI routes the case to your manual cropping script before classification.
                    </p>
                  </div>
                  <Switch id="external-parts" checked={hasExternalChest} onCheckedChange={setHasExternalChest} />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="notes">Run notes</Label>
                  <Textarea
                    id="notes"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Optional notes for the exported report"
                    className="min-h-24"
                  />
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button onClick={runPipeline} disabled={isRunning}>
                    {isRunning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                    Run pipeline
                  </Button>
                  <Button variant="outline" onClick={resetPipeline} disabled={isRunning}>
                    Reset
                  </Button>
                  <Button variant="outline" onClick={exportReport} disabled={!outputs?.reportRows?.length}>
                    <Download className="mr-2 h-4 w-4" />
                    Export report
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        <div className="space-y-6">
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <Card className="rounded-2xl border-0 shadow-sm">
              <CardHeader>
                <CardTitle>2. Pipeline execution status</CardTitle>
                <CardDescription>Track each stage of the workflow from validation to report generation.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-600">Overall progress</span>
                    <span className="font-medium">{progress}%</span>
                  </div>
                  <Progress value={progress} />
                </div>

                <div className="space-y-3">
                  {STEP_META.map((step, index) => {
                    const Icon = step.icon;
                    return (
                      <div key={step.key} className="rounded-2xl border bg-white p-4">
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex gap-3">
                            <div className="mt-0.5 rounded-xl bg-slate-100 p-2">
                              <Icon className="h-4 w-4" />
                            </div>
                            <div>
                              <div className="font-medium">{index + 1}. {step.title}</div>
                              <div className="text-sm text-slate-500">{step.desc}</div>
                            </div>
                          </div>
                          <StatusBadge status={stepStatuses[step.key]} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
            <Card className="rounded-2xl border-0 shadow-sm">
              <CardHeader>
                <CardTitle>3. Mock outputs</CardTitle>
                <CardDescription>
                  Current values are placeholder frontend outputs so your team can review the user journey before the backend is connected.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {!outputs ? (
                  <div className="rounded-2xl border border-dashed p-8 text-center text-sm text-slate-500">
                    No pipeline outputs yet. Upload a valid case and run the prototype.
                  </div>
                ) : (
                  <>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div className="rounded-2xl border bg-white p-4">
                        <div className="text-sm text-slate-500">Lesion screening</div>
                        <div className="mt-1 text-lg font-semibold">{outputs.summary.lesionLabel}</div>
                        <div className="text-sm text-slate-600">Probability: {(outputs.summary.lesionProbability * 100).toFixed(1)}%</div>
                      </div>
                      <div className="rounded-2xl border bg-white p-4">
                        <div className="text-sm text-slate-500">Subtype classification</div>
                        <div className="mt-1 text-lg font-semibold">{outputs.summary.lesionTypeLabel}</div>
                        <div className="text-sm text-slate-600">
                          Probability: {outputs.summary.lesionTypeProbability !== null ? `${(outputs.summary.lesionTypeProbability * 100).toFixed(1)}%` : "N/A"}
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border bg-white p-4">
                      <div className="text-sm text-slate-500">Segmentation result</div>
                      <div className="mt-1 text-lg font-semibold">{outputs.summary.segmentationStatus}</div>
                      <div className="text-sm text-slate-600">Mask file: {outputs.summary.maskFilename}</div>
                    </div>

                    <Separator />

                    <div className="rounded-2xl border bg-white p-4">
                      <div className="mb-2 font-medium">Preview of exported report</div>
                      <pre className="overflow-x-auto rounded-xl bg-slate-50 p-3 text-xs leading-6 text-slate-700">
                        {JSON.stringify(outputs.reportRows[0], null, 2)}
                      </pre>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
