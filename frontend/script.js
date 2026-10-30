// ============================================================================
// CONFIGURATION
// ----------------------------------------------------------------------------
// API_BASE_URL is the address of the FastAPI backend.
//
// Local development: "http://localhost:8000" (the default below).
//
// AFTER you deploy the backend to Hugging Face Spaces, replace this value
// with your Spaces backend URL, for example:
//   const API_BASE_URL = "https://<username>-groundedqa.hf.space";
// ============================================================================
const API_BASE_URL = "http://localhost:8000";

// Sample from the project README (NIT Patna example).
const SAMPLE_CONTEXT =
  "The National Institute of Technology Patna is a public technical university located in Patna, Bihar, India.";
const SAMPLE_QUESTION =
  "Where is the National Institute of Technology Patna located?";

// --- DOM elements -----------------------------------------------------------
const contextEl = document.getElementById("context");
const questionEl = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const sampleBtn = document.getElementById("sample-btn");
const errorEl = document.getElementById("error");
const resultCard = document.getElementById("result-card");
const answerEl = document.getElementById("answer");
const scoreEl = document.getElementById("score");
const confidenceLine = document.getElementById("confidence-line");
const contextBlock = document.getElementById("context-block");
const contextViewEl = document.getElementById("context-view");

// --- Helpers ----------------------------------------------------------------
function getSelectedMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "extractive";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}

function setLoading(isLoading) {
  askBtn.disabled = isLoading;
  askBtn.textContent = isLoading ? "Finding answer..." : "Get Answer";
}

// Render the context with the answer span wrapped in <mark> (extractive mode).
function renderHighlightedContext(context, start, end) {
  const before = escapeHtml(context.slice(0, start));
  const answer = escapeHtml(context.slice(start, end));
  const after = escapeHtml(context.slice(end));
  contextViewEl.innerHTML = `${before}<mark>${answer}</mark>${after}`;
}

// --- Result rendering ---------------------------------------------------------
function showExtractiveResult(data, context) {
  answerEl.textContent = data.answer || "(no answer found)";
  answerEl.classList.remove("answer-notice");
  scoreEl.textContent = `${(data.score * 100).toFixed(1)}%`;
  confidenceLine.classList.remove("hidden");
  renderHighlightedContext(context, data.start, data.end);
  contextBlock.classList.remove("hidden");
  resultCard.classList.remove("hidden");
}

function showGenaiResult(data) {
  answerEl.textContent = data.answer || "(no answer returned)";
  // No span highlight or confidence score in generative mode.
  confidenceLine.classList.add("hidden");
  contextBlock.classList.add("hidden");
  // Out-of-context refusal (grounded = false) is shown as a notice.
  answerEl.classList.toggle("answer-notice", data.grounded === false);
  resultCard.classList.remove("hidden");
}

// --- Actions ------------------------------------------------------------------
sampleBtn.addEventListener("click", () => {
  contextEl.value = SAMPLE_CONTEXT;
  questionEl.value = SAMPLE_QUESTION;
  hideError();
});

askBtn.addEventListener("click", getAnswer);
questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") getAnswer();
});

async function getAnswer() {
  const context = contextEl.value.trim();
  const question = questionEl.value.trim();
  const mode = getSelectedMode();

  hideError();
  resultCard.classList.add("hidden");

  if (!context || !question) {
    showError("Please enter both a paragraph and a question.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context, question, mode }),
    });

    if (!response.ok) {
      let detail = `Server returned ${response.status}`;
      try {
        const data = await response.json();
        if (data.detail) detail = data.detail;
      } catch (_) {
        /* keep default message */
      }
      throw new Error(detail);
    }

    const data = await response.json();
    if (data.mode === "genai") {
      showGenaiResult(data);
    } else {
      showExtractiveResult(data, context);
    }
  } catch (err) {
    showError(
      `Could not get an answer: ${err.message}. ` +
        "Please check that the backend is running and try again."
    );
  } finally {
    setLoading(false);
  }
}
