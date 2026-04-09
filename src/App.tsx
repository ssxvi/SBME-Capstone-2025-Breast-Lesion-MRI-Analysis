import { useEffect, useMemo, useRef, useState } from "react";
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
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type InputType = "nifti" | "dicom" | "invalid" | null;
type StepStatusValue = "pending" | "running" | "complete" | "skipped" | "error";
type JobStatusValue = "queued" | "running" | "completed" | "failed" | "cancelled";
type StepKey = "validate" | "convert" | "crop" | "lesionDetection" | "lesionType" | "segmentation" | "report";

type StepStatuses = Record<StepKey, StepStatusValue>;

type UploadResponse = {
  upload_id: string;
  detected_input_type: "nifti" | "dicom" | "invalid";
  file_count: number;
  validation: {
    ok: boolean;
    message: string;
  };
};

type CreateJobResponse = {
  job_id: string;
  status: JobStatusValue;
};

type JobStatusResponse = {
  job_id: string;
  status: JobStatusValue;
  progress: number;
  steps: StepStatuses;
  message: string;
};

type ReportRow = {
  pipeline_name: string;
  uploaded_input_type: string;
  uploaded_files: string;
  external_chest_present: string;
  dicom_to_nifti: string;
  manual_cropping: string;
  lesion_screening_label: string;
  lesion_screening_probability: string;
  lesion_type_label: string;
  lesion_type_probability: string;
  segmentation_result: string;
  segmentation_mask: string;
  notes: string;
};

type JobResultResponse = {
  job_id: string;
  status: JobStatusValue;
  summary: {
    lesionLabel: string;
    lesionProbability: number;
    lesionTypeLabel: string;
    lesionTypeProbability: number | null;
    segmentationStatus: string;
    maskFilename: string;
  };
  report_row: ReportRow;
};

type UiOutputs = {
  summary: JobResultResponse["summary"];
  reportRows: ReportRow[];
};

const defaultStepStatuses = (): StepStatuses => ({
  validate: "pending",
  convert: "pending",
  crop: "pending",
  lesionDetection: "pending",
  lesionType: "pending",
  segmentation: "pending",
  report: "pending",
});

function detectInputType(files: File[]): InputType {
  if (!files?.length) return null;

  const names = Array.from(files).map((f) => f.name.toLowerCase());
  const hasNifti = names.some((n) => n.endsWith(".nii") || n.endsWith(".nii.gz"));
  const allDicom = names.every((n) => n.endsWith(".dcm"));

  if (hasNifti) return "nifti";
  if (allDicom) return "dicom";
  return "invalid";
}

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function StatusBadge({ status }: { status: StepStatusValue }) {
  if (status === "complete") return <Badge className="gap-1 flex items-center bg-green-50 text-green-700 border-green-200"><CheckCircle2 className="h-3.5 w-3.5" />Complete</Badge>;
  if (status === "running") return <Badge variant="secondary" className="gap-1 flex items-center"><Loader2 className="h-3.5 w-3.5 animate-spin" />Running</Badge>;
  if (status === "skipped") return <Badge variant="outline">Skipped</Badge>;
  if (status === "error") return <Badge variant="destructive" className="gap-1 flex items-center"><AlertCircle className="h-3.5 w-3.5" />Error</Badge>;
  return <Badge variant="outline" className="bg-slate-50">Pending</Badge>;
}

function getIconBgColor(status: StepStatusValue): string {
  if (status === "complete") return "bg-green-100";
  if (status === "running") return "bg-blue-100";
  if (status === "error") return "bg-red-100";
  if (status === "skipped") return "bg-slate-100";
  return "bg-slate-50";
}

export default function BreastImagingPipelineUI() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [inputType, setInputType] = useState<InputType>(null);
  const [validationMessage, setValidationMessage] = useState("No files uploaded yet.");
  const [hasExternalChest, setHasExternalChest] = useState(false);
  const [reportFormat, setReportFormat] = useState("csv");
  const [pipelineName, setPipelineName] = useState("Breast Lesion MRI/Ultrasound Pipeline");
  const [notes, setNotes] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stepStatuses, setStepStatuses] = useState<StepStatuses>(defaultStepStatuses());
  const [outputs, setOutputs] = useState<UiOutputs | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      setIsRunning(false);
    };
  }, []);

  const fileSummary = useMemo(() => {
    if (!files.length) return "No files selected";
    if (inputType === "dicom") return `${files.length} DICOM file${files.length > 1 ? "s" : ""} selected`;
    if (inputType === "nifti") return `${files.length} NIfTI file${files.length > 1 ? "s" : ""} selected`;
    return `${files.length} file${files.length > 1 ? "s" : ""} selected`;
  }, [files, inputType]);

  const handleFiles = (fileList: FileList | File[]) => {
    const nextFiles = Array.from(fileList || []);
    setFiles(nextFiles);
    setOutputs(null);
    setJobId(null);
    setProgress(0);
    setStepStatuses(defaultStepStatuses());

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

  const markStep = (key: StepKey, status: StepStatusValue) => {
    setStepStatuses((prev) => ({ ...prev, [key]: status }));
  };

  const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const resetPipeline = () => {
    setOutputs(null);
    setJobId(null);
    setProgress(0);
    setIsRunning(false);
    setStepStatuses(defaultStepStatuses());
  };

  const parseError = async (response: Response) => {
    try {
      const body = await response.json() as { detail?: string };
      return body?.detail || `Request failed (${response.status})`;
    } catch {
      if (response.status === 0 || response.statusText === "error") {
        return "Connection error: Unable to reach the server. Please check your internet connection and try again.";
      }
      return `Request failed with status ${response.status}`;
    }
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
    setStepStatuses(defaultStepStatuses());

    try {
      markStep("validate", "running");
      setProgress(10);

      // Upload the first 3 files as Pre, Post_1, Post_2
      // (Backend expects exactly 3 timepoints)
      const filesToUpload = files.slice(0, 3);
      if (filesToUpload.length < 3) {
        throw new Error("Please upload at least 3 timepoint files (Pre, Post_1, Post_2)");
      }

      const uploadedPaths: string[] = [];
      const timepoints = ["Pre", "Post_1", "Post_2"];

      for (let i = 0; i < filesToUpload.length; i++) {
        const file = filesToUpload[i];
        const formData = new FormData();
        formData.append("file", file);

        setProgress(10 + (i * 20) / 3);
        const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error(await parseError(uploadResponse));
        }

        const uploadData = await uploadResponse.json() as { server_path: string };
        uploadedPaths.push(uploadData.server_path);
      }

      markStep("validate", "complete");
      setProgress(30);
      setValidationMessage(
        `Successfully uploaded ${filesToUpload.length} files. Starting pipeline analysis...`
      );

      // Submit pipeline job
      markStep("convert", "running");
      setProgress(35);

      const runResponse = await fetch(`${API_BASE_URL}/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          case_id: pipelineName,
          pre_path: uploadedPaths[0],
          post1_path: uploadedPaths[1],
          post2_path: uploadedPaths[2],
          skip_segmentation: false,
          use_mask_for_roi: true,
          lesion_threshold: 0.5,
          nnunet_dataset_id: "001",
          nnunet_configuration: "3d_fullres",
          nnunet_fold: "0",
        }),
      });

      if (!runResponse.ok) {
        throw new Error(await parseError(runResponse));
      }

      const runData = await runResponse.json() as { run_id: string; status: string };
      setJobId(runData.run_id);
      setProgress(40);

      // Poll for results
      let pipelineResult: any = null;
      let pollCount = 0;
      const maxPolls = 600; // ~10 minutes at 1s intervals

      while (pollCount < maxPolls) {
        const resultResponse = await fetch(`${API_BASE_URL}/result/${runData.run_id}`);
        if (!resultResponse.ok) {
          throw new Error(await parseError(resultResponse));
        }

        pipelineResult = await resultResponse.json();
        const status = pipelineResult.status;

        console.log(`[Poll ${pollCount}] Status: ${status}`, pipelineResult);

        // Mark convert as running/complete (DICOM→NIfTI conversion happens early)
        if (status === "running" || status === "Running") {
          markStep("convert", "running");
          markStep("crop", "running");
          setProgress(45);
        }

        // Update progress and steps based on results received
        if (pipelineResult.lesion) {
          markStep("convert", "complete");
          markStep("crop", "complete");
          markStep("lesionDetection", "complete");
          setProgress(60);
        }

        if (pipelineResult.malignancy) {
          markStep("lesionType", "complete");
          setProgress(75);
        } else if (pipelineResult.lesion && (status === "running" || status === "Running")) {
          // Lesion found but malignancy still processing
          markStep("lesionType", "running");
        }

        if (pipelineResult.segmentation) {
          markStep("segmentation", "complete");
          setProgress(85);
        } else if (pipelineResult.lesion && (status === "running" || status === "Running")) {
          // Lesion found but segmentation still processing
          markStep("segmentation", "running");
        }

        // Check for terminal states
        if (status === "complete" || status === "Complete") {
          markStep("convert", "complete");
          markStep("crop", "complete");
          markStep("lesionDetection", "complete");
          if (pipelineResult.malignancy) markStep("lesionType", "complete");
          if (pipelineResult.segmentation) markStep("segmentation", "complete");
          markStep("report", "complete");
          setProgress(100);
          console.log("Pipeline complete!");
          break;
        }

        if (status === "failed" || status === "Failed") {
          throw new Error(pipelineResult.error || "Pipeline failed");
        }

        await wait(1000);
        pollCount++;
      }

      if (pollCount >= maxPolls) {
        throw new Error("Pipeline timeout: exceeded maximum polling attempts");
      }

      // Transform backend response to UI format
      const summary = {
        lesionLabel: pipelineResult.lesion?.label || "unknown",
        lesionProbability: pipelineResult.lesion?.confidence || 0,
        lesionTypeLabel: pipelineResult.malignancy?.label || null,
        lesionTypeProbability: pipelineResult.malignancy?.confidence || null,
        segmentationStatus: pipelineResult.segmentation ? "Complete" : "Skipped",
        maskFilename: pipelineResult.segmentation?.mask_path?.split("/").pop() || "N/A",
      };

      const reportRow: ReportRow = {
        pipeline_name: pipelineName,
        uploaded_input_type: inputType || "unknown",
        uploaded_files: filesToUpload.map((f) => f.name).join("; "),
        external_chest_present: hasExternalChest ? "Yes" : "No",
        dicom_to_nifti: inputType === "dicom" ? "Yes" : "No",
        manual_cropping: hasExternalChest ? "Yes" : "No",
        lesion_screening_label: pipelineResult.lesion?.label || "N/A",
        lesion_screening_probability: `${((pipelineResult.lesion?.confidence || 0) * 100).toFixed(1)}%`,
        lesion_type_label: pipelineResult.malignancy?.label || "N/A",
        lesion_type_probability: pipelineResult.malignancy
          ? `${(pipelineResult.malignancy.confidence * 100).toFixed(1)}%`
          : "N/A",
        segmentation_result: pipelineResult.segmentation ? "Success" : "Skipped",
        segmentation_mask: pipelineResult.segmentation?.mask_path || "N/A",
        notes,
      };

      const nextOutputs: UiOutputs = {
        summary,
        reportRows: [reportRow],
      };

      setOutputs(nextOutputs);
      setValidationMessage("Pipeline analysis complete!");
    } catch (error) {
      console.error(error);
      let message = "Pipeline execution failed.";

      if (error instanceof Error) {
        if (error.message.includes("fetch")) {
          message = "Unable to connect to the pipeline server. Please ensure the backend is running and try again.";
        } else if (error.message.includes("timeout")) {
          message = "Pipeline took too long to complete. Please try again or contact support if the issue persists.";
        } else if (error.message.includes("at least 3")) {
          message = error.message;
        } else {
          message = error.message;
        }
      }

      setValidationMessage(message);
      markStep("validate", "error");
    } finally {
      setIsRunning(false);
    }
  };

  const exportReport = async () => {
    if (!outputs?.reportRows?.length) return;

    try {
      const reportRow = outputs.reportRows[0];

      if (reportFormat === "csv") {
        // Convert to CSV
        const headers = Object.keys(reportRow);
        const values = Object.values(reportRow);
        const csvContent = [
          headers.join(","),
          values.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","),
        ].join("\n");

        const blob = new Blob([csvContent], { type: "text/csv" });
        downloadBlob("pipeline_report.csv", blob);
      } else {
        // Export as JSON
        const jsonContent = JSON.stringify([reportRow], null, 2);
        const blob = new Blob([jsonContent], { type: "application/json" });
        downloadBlob("pipeline_report.json", blob);
      }
    } catch (error) {
      console.error(error);
      let message = "Export failed.";
      if (error instanceof Error) {
        message = error.message.includes("blob")
          ? "Failed to create report file. Try again or contact support."
          : error.message;
      }
      setValidationMessage(message);
    }
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
                  Please upload 3 T1 MRI images in order of Pre, Post1, and Post2. 
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
                      onChange={(e) => handleFiles(e.target.files || [])}
                    />
                    <div className="flex flex-col items-center justify-center gap-3 text-center">
                      <div className="rounded-full bg-slate-100 p-4">
                        <Upload className="h-6 w-6" />
                      </div>
                      <div>
                        <div className="font-medium">Upload NIfTI file or DICOM series</div>
                        <div className="text-sm text-slate-600">For DICOM, select all files from one series together.</div>
                      </div>
                      <div className="flex flex-wrap justify-center gap-3">
                        <Button onClick={() => inputRef.current?.click()} className="flex items-center">
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
                  <Button onClick={runPipeline} disabled={isRunning} className="flex items-center">
                    {isRunning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                    Run pipeline
                  </Button>
                  <Button variant="outline" onClick={resetPipeline} disabled={isRunning}>
                    Reset
                  </Button>
                  <Button variant="outline" onClick={exportReport} disabled={!outputs?.reportRows?.length} className="flex items-center">
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
                    <span className="font-semibold text-lg text-blue-600">{progress}%</span>
                  </div>
                  <Progress value={progress} />
                </div>

                <div className="space-y-3">
                  {STEP_META.map((step, index) => {
                    const Icon = step.icon;
                    const stepKey = step.key as StepKey;
                    const status = stepStatuses[stepKey];
                    const iconBgColor = getIconBgColor(status);
                    return (
                      <div key={step.key} className={`rounded-2xl border p-4 transition-colors ${status === "pending" ? "bg-white" : "bg-white"}`}>
                        <div className="flex items-center justify-between gap-4">
                          <div className="flex items-center gap-3">
                            <div className={`rounded-xl p-2 ${iconBgColor}`}>
                              <Icon className="h-4 w-4" />
                            </div>
                            <div className="flex-1">
                              <div className="font-medium">{index + 1}. {step.title}</div>
                              <div className="text-xs text-slate-600">{step.desc}</div>
                            </div>
                          </div>
                          <StatusBadge status={status} />
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
                <CardTitle>3. Report Selector</CardTitle>
                <CardDescription>
                  Current values are placeholder frontend outputs so your team can review the user journey before the backend is connected.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {!outputs ? (
                  <div className="rounded-2xl border border-dashed p-8 text-center text-sm text-slate-600">
                    No pipeline outputs yet. Upload a valid case and run the prototype.
                  </div>
                ) : (
                  <>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div className="rounded-2xl border bg-white p-4">
                        <div className="text-sm text-slate-600">Lesion screening</div>
                        <div className="mt-1 text-lg font-semibold">{outputs.summary.lesionLabel}</div>
                        <div className="text-sm text-slate-600">Probability: {(outputs.summary.lesionProbability * 100).toFixed(1)}%</div>
                      </div>
                      <div className="rounded-2xl border bg-white p-4">
                        <div className="text-sm text-slate-600">Subtype classification</div>
                        <div className="mt-1 text-lg font-semibold">{outputs.summary.lesionTypeLabel}</div>
                        <div className="text-sm text-slate-600">
                          Probability: {outputs.summary.lesionTypeProbability !== null ? `${(outputs.summary.lesionTypeProbability * 100).toFixed(1)}%` : "N/A"}
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border bg-white p-4">
                      <div className="text-sm text-slate-600">Segmentation result</div>
                      <div className="mt-1 text-lg font-semibold">{outputs.summary.segmentationStatus}</div>
                      <div className="text-sm text-slate-600">Mask file: {outputs.summary.maskFilename}</div>
                    </div>

                    <Separator />

                    <div className="flex gap-2">
                      <Button
                        onClick={() => {
                          if (jobId) {
                            window.open(`${API_BASE_URL}/report/${jobId}`, "_blank");
                          }
                        }}
                        className="flex-1 bg-blue-600 hover:bg-blue-700"
                      >
                        <FileText className="h-4 w-4 mr-2" />
                        Open HTML Report
                      </Button>
                      <Button
                        onClick={() => {
                          const csv = [
                            Object.keys(outputs.reportRows[0]).join(","),
                            ...outputs.reportRows.map(row =>
                              Object.values(row).map(v =>
                                typeof v === "string" && v.includes(",") ? `"${v}"` : v
                              ).join(",")
                            ),
                          ].join("\n");
                          downloadBlob(`report_${new Date().toISOString().split("T")[0]}.csv`, new Blob([csv], { type: "text/csv" }));
                        }}
                        variant="outline"
                        className="flex-1"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download CSV
                      </Button>
                    </div>

                    <div className="rounded-2xl border bg-white p-4">
                      <div className="mb-2 font-medium">Preview of exported report</div>
                      <pre className="overflow-x-auto rounded-xl bg-slate-100 p-3 text-xs leading-6 text-slate-700">
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
