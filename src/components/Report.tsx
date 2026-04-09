export interface ReportData {
  id: string;
  timestamp: Date;
  pipelineName: string;
  lesionDetection: {
    label: string;
    probability: number;
  };
  malignancyClassification: {
    label: string | null;
    probability: number | null;
  };
  segmentation: {
    status: string;
    maskFilename: string;
  };
  notes?: string;
}

interface ReportProps {
  data: ReportData;
  onClose?: () => void;
}

const colorScheme = {
  danger: "#e05c5c",
  dangerLight: "#2e1a1a",
  dangerDark: "#c94a4a",
  warning: "#e0a94a",
  warningLight: "#2e1f1a",
  warningDark: "#c49540",
  success: "#4caf82",
  successLight: "#1a2e1f",
  benign: "#7ac46a",
  benignLight: "#1f2a1a",
  accent: "#6c63ff",
  accentLight: "#1a1a2e",
  textPrimary: "#e2e2f0",
  textSecondary: "#8888aa",
  bgPrimary: "#1a1a2e",
  bgSecondary: "#2a2a4a",
  bgTertiary: "#2a2a4a",
  bg: "#0f0f1a",
};

export function Report({ data, onClose }: ReportProps) {
  const lesionConfidence = (data.lesionDetection.probability * 100).toFixed(1);
  const malignancyConfidence =
    data.malignancyClassification.probability !== null
      ? (data.malignancyClassification.probability * 100).toFixed(1)
      : null;

  const lesionGaugeOffset =
    (data.lesionDetection.probability / 1) * 197.9 - 197.9;
  const malignancyGaugeOffset =
    data.malignancyClassification.probability !== null
      ? (data.malignancyClassification.probability / 1) * 197.9 - 197.9
      : 0;

  return (
    <div
      style={{
        fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
        backgroundColor: "#0f0f1a",
        minHeight: "100vh",
        padding: "2rem",
        color: colorScheme.textPrimary,
      }}
    >
      <div
        style={{
          backgroundColor: colorScheme.bgPrimary,
          maxWidth: "860px",
          margin: "0 auto",
          borderRadius: "12px",
          boxShadow: "0 4px 32px rgba(0,0,0,0.30), 0 1px 4px rgba(0,0,0,0.20)",
          padding: "2.5rem 2.75rem",
          border: `1px solid ${colorScheme.bgSecondary}`,
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            paddingBottom: "1.25rem",
            borderBottom: `1px solid ${colorScheme.bgSecondary}`,
            marginBottom: "1.75rem",
          }}
        >
          <div>
            <div
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "10px",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                color: colorScheme.textSecondary,
                marginBottom: "4px",
              }}
            >
              Breast MRI Analysis Report
            </div>
            <div style={{ fontSize: "20px", fontWeight: 500, marginBottom: "3px" }}>
              {data.pipelineName}
            </div>
            <div style={{ fontSize: "13px", color: colorScheme.textSecondary }}>
              {data.timestamp.toLocaleString()}
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", alignItems: "flex-end" }}>
            {data.lesionDetection.label === "Positive" && (
              <span
                style={{
                  backgroundColor: colorScheme.dangerLight,
                  color: colorScheme.danger,
                  padding: "3px 9px",
                  borderRadius: "3px",
                  fontSize: "10px",
                  fontWeight: 500,
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                  border: `1px solid ${colorScheme.danger}`,
                }}
              >
                Lesion Detected
              </span>
            )}
            {data.malignancyClassification.label && (
              <span
                style={{
                  backgroundColor:
                    data.malignancyClassification.label === "Malignant"
                      ? colorScheme.warningLight
                      : colorScheme.successLight,
                  color:
                    data.malignancyClassification.label === "Malignant"
                      ? colorScheme.warning
                      : colorScheme.success,
                  padding: "3px 9px",
                  borderRadius: "3px",
                  fontSize: "10px",
                  fontWeight: 500,
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                  border: `1px solid ${
                    data.malignancyClassification.label === "Malignant"
                      ? colorScheme.warning
                      : colorScheme.success
                  }`,
                }}
              >
                {data.malignancyClassification.label}
              </span>
            )}
          </div>
        </div>

        {/* Detection Confidence Section */}
        <div style={{ marginBottom: "1.75rem" }}>
          <div
            style={{
              fontSize: "10px",
              textTransform: "uppercase",
              letterSpacing: "0.13em",
              color: colorScheme.textSecondary,
              fontWeight: 500,
              marginBottom: "0.7rem",
            }}
          >
            Detection confidence
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
            {/* Lesion Detection Card */}
            <div
              style={{
                backgroundColor: colorScheme.bgPrimary,
                border: `1px solid ${colorScheme.bgSecondary}`,
                borderRadius: "12px",
                padding: "1.1rem 1.25rem",
              }}
            >
              <div style={{ marginBottom: "12px", display: "flex", justifyContent: "space-between" }}>
                <div style={{ fontSize: "13px", fontWeight: 500 }}>Lesion presence</div>
                <span
                  style={{
                    backgroundColor: colorScheme.dangerLight,
                    color: colorScheme.danger,
                    padding: "3px 9px",
                    borderRadius: "3px",
                    fontSize: "10px",
                    fontWeight: 500,
                    textTransform: "uppercase",
                    border: `1px solid ${colorScheme.danger}`,
                  }}
                >
                  {data.lesionDetection.label}
                </span>
              </div>

              <svg width="150" height="84" viewBox="0 0 150 84" style={{ margin: "0 auto", display: "block" }}>
                <path
                  d="M 12 74 A 63 63 0 0 1 138 74"
                  fill="none"
                  stroke={colorScheme.bgSecondary}
                  strokeWidth="11"
                  strokeLinecap="round"
                />
                <path
                  d="M 12 74 A 63 63 0 0 1 138 74"
                  fill="none"
                  stroke={colorScheme.danger}
                  strokeWidth="11"
                  strokeLinecap="round"
                  strokeDasharray="197.9"
                  strokeDashoffset={lesionGaugeOffset}
                />
                <text
                  x="75"
                  y="68"
                  textAnchor="middle"
                  fontSize="21"
                  fontWeight="500"
                  fontFamily="IBM Plex Mono, monospace"
                  fill={colorScheme.textPrimary}
                >
                  {lesionConfidence}%
                </text>
                <text
                  x="75"
                  y="80"
                  textAnchor="middle"
                  fontSize="10"
                  fill={colorScheme.textSecondary}
                >
                  confidence
                </text>
              </svg>

              <div style={{ height: "1px", backgroundColor: colorScheme.bgSecondary, margin: "0.9rem 0" }} />

              <div style={{ marginBottom: "7px", display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ fontSize: "12px", color: colorScheme.textSecondary, width: "90px" }}>
                  Lesion
                </span>
                <div
                  style={{
                    flex: 1,
                    height: "6px",
                    backgroundColor: colorScheme.bgSecondary,
                    borderRadius: "3px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      borderRadius: "3px",
                      backgroundColor: colorScheme.danger,
                      width: `${data.lesionDetection.probability * 100}%`,
                    }}
                  />
                </div>
                <span
                  style={{
                    fontSize: "12px",
                    fontFamily: "'IBM Plex Mono', monospace",
                    color: colorScheme.textPrimary,
                    width: "36px",
                    textAlign: "right",
                  }}
                >
                  {lesionConfidence}%
                </span>
              </div>

              <div style={{ marginBottom: "7px", display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ fontSize: "12px", color: colorScheme.textSecondary, width: "90px" }}>
                  No lesion
                </span>
                <div
                  style={{
                    flex: 1,
                    height: "6px",
                    backgroundColor: colorScheme.bgSecondary,
                    borderRadius: "3px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      borderRadius: "3px",
                      backgroundColor: colorScheme.success,
                      width: `${(1 - data.lesionDetection.probability) * 100}%`,
                    }}
                  />
                </div>
                <span
                  style={{
                    fontSize: "12px",
                    fontFamily: "'IBM Plex Mono', monospace",
                    color: colorScheme.textPrimary,
                    width: "36px",
                    textAlign: "right",
                  }}
                >
                  {(100 - parseFloat(lesionConfidence)).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Malignancy Classification Card */}
            {data.malignancyClassification.label && malignancyConfidence && (
              <div
                style={{
                  backgroundColor: colorScheme.bgPrimary,
                  border: `1px solid ${colorScheme.bgSecondary}`,
                  borderRadius: "12px",
                  padding: "1.1rem 1.25rem",
                }}
              >
                <div style={{ marginBottom: "12px", display: "flex", justifyContent: "space-between" }}>
                  <div style={{ fontSize: "13px", fontWeight: 500 }}>Malignancy classification</div>
                  <span
                    style={{
                      backgroundColor:
                        data.malignancyClassification.label === "Malignant"
                          ? colorScheme.warningLight
                          : colorScheme.successLight,
                      color:
                        data.malignancyClassification.label === "Malignant"
                          ? colorScheme.warning
                          : colorScheme.success,
                      padding: "3px 9px",
                      borderRadius: "3px",
                      fontSize: "10px",
                      fontWeight: 500,
                      textTransform: "uppercase",
                      border: `1px solid ${
                        data.malignancyClassification.label === "Malignant"
                          ? colorScheme.warning
                          : colorScheme.success
                      }`,
                    }}
                  >
                    {data.malignancyClassification.label}
                  </span>
                </div>

                <svg width="150" height="84" viewBox="0 0 150 84" style={{ margin: "0 auto", display: "block" }}>
                  <path
                    d="M 12 74 A 63 63 0 0 1 138 74"
                    fill="none"
                    stroke={colorScheme.bgSecondary}
                    strokeWidth="11"
                    strokeLinecap="round"
                  />
                  <path
                    d="M 12 74 A 63 63 0 0 1 138 74"
                    fill="none"
                    stroke={
                      data.malignancyClassification.label === "Malignant" ? colorScheme.warning : colorScheme.success
                    }
                    strokeWidth="11"
                    strokeLinecap="round"
                    strokeDasharray="197.9"
                    strokeDashoffset={malignancyGaugeOffset}
                  />
                  <text
                    x="75"
                    y="68"
                    textAnchor="middle"
                    fontSize="21"
                    fontWeight="500"
                    fontFamily="IBM Plex Mono, monospace"
                    fill={colorScheme.textPrimary}
                  >
                    {malignancyConfidence}%
                  </text>
                  <text
                    x="75"
                    y="80"
                    textAnchor="middle"
                    fontSize="10"
                    fill={colorScheme.textSecondary}
                  >
                    confidence
                  </text>
                </svg>

                <div style={{ height: "1px", backgroundColor: colorScheme.bgSecondary, margin: "0.9rem 0" }} />

                <div style={{ marginBottom: "7px", display: "flex", alignItems: "center", gap: "10px" }}>
                  <span style={{ fontSize: "12px", color: colorScheme.textSecondary, width: "90px" }}>
                    {data.malignancyClassification.label}
                  </span>
                  <div
                    style={{
                      flex: 1,
                      height: "6px",
                      backgroundColor: colorScheme.bgSecondary,
                      borderRadius: "3px",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        borderRadius: "3px",
                        backgroundColor:
                          data.malignancyClassification.label === "Malignant" ? colorScheme.warning : colorScheme.benign,
                        width: `${data.malignancyClassification.probability! * 100}%`,
                      }}
                    />
                  </div>
                  <span
                    style={{
                      fontSize: "12px",
                      fontFamily: "'IBM Plex Mono', monospace",
                      color: colorScheme.textPrimary,
                      width: "36px",
                      textAlign: "right",
                    }}
                  >
                    {malignancyConfidence}%
                  </span>
                </div>

                <div style={{ marginBottom: "7px", display: "flex", alignItems: "center", gap: "10px" }}>
                  <span style={{ fontSize: "12px", color: colorScheme.textSecondary, width: "90px" }}>
                    {data.malignancyClassification.label === "Malignant" ? "Benign" : "Malignant"}
                  </span>
                  <div
                    style={{
                      flex: 1,
                      height: "6px",
                      backgroundColor: colorScheme.bgSecondary,
                      borderRadius: "3px",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        borderRadius: "3px",
                        backgroundColor:
                          data.malignancyClassification.label === "Malignant" ? colorScheme.benign : colorScheme.warning,
                        width: `${(1 - data.malignancyClassification.probability!) * 100}%`,
                      }}
                    />
                  </div>
                  <span
                    style={{
                      fontSize: "12px",
                      fontFamily: "'IBM Plex Mono', monospace",
                      color: colorScheme.textPrimary,
                      width: "36px",
                      textAlign: "right",
                    }}
                  >
                    {(100 - parseFloat(malignancyConfidence)).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Segmentation Section */}
        {data.segmentation && (
          <div style={{ marginBottom: "1.75rem" }}>
            <div
              style={{
                fontSize: "10px",
                textTransform: "uppercase",
                letterSpacing: "0.13em",
                color: colorScheme.textSecondary,
                fontWeight: 500,
                marginBottom: "0.7rem",
              }}
            >
              Segmentation result
            </div>
            <div
              style={{
                backgroundColor: colorScheme.bgPrimary,
                border: `1px solid ${colorScheme.bgSecondary}`,
                borderRadius: "12px",
                padding: "1.1rem 1.25rem",
              }}
            >
              <div style={{ fontSize: "13px", fontWeight: 500, marginBottom: "0.5rem" }}>
                {data.segmentation.status}
              </div>
              <div style={{ fontSize: "12px", color: colorScheme.textSecondary }}>
                Mask file: {data.segmentation.maskFilename}
              </div>
            </div>
          </div>
        )}

        {/* Notes Section */}
        {data.notes && (
          <div style={{ marginBottom: "1.75rem" }}>
            <div
              style={{
                fontSize: "10px",
                textTransform: "uppercase",
                letterSpacing: "0.13em",
                color: colorScheme.textSecondary,
                fontWeight: 500,
                marginBottom: "0.7rem",
              }}
            >
              Notes
            </div>
            <div
              style={{
                backgroundColor: colorScheme.bgSecondary,
                borderLeft: `2px solid ${colorScheme.danger}`,
                borderRadius: "0 8px 8px 0",
                padding: "0.6rem 0.9rem",
                fontSize: "12px",
                color: colorScheme.textSecondary,
                lineHeight: 1.6,
              }}
            >
              {data.notes}
            </div>
          </div>
        )}

        {/* Footer */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            paddingTop: "1.1rem",
            borderTop: `1px solid ${colorScheme.bgSecondary}`,
            marginTop: "1.25rem",
          }}
        >
          <span style={{ fontSize: "11px", color: colorScheme.textSecondary, fontFamily: "'IBM Plex Mono', monospace" }}>
            For clinical review only — not a diagnostic substitute
          </span>
          {onClose && (
            <button
              onClick={onClose}
              style={{
                padding: "8px 22px",
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "12px",
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                backgroundColor: colorScheme.bgPrimary,
                border: `1px solid ${colorScheme.bgSecondary}`,
                borderRadius: "8px",
                color: colorScheme.textSecondary,
                cursor: "pointer",
                transition: "background 0.15s",
              }}
              onMouseEnter={(e) => {
                (e.target as HTMLButtonElement).style.backgroundColor = colorScheme.bgSecondary;
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLButtonElement).style.backgroundColor = colorScheme.bgPrimary;
              }}
            >
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
