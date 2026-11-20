import React, { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ApiResponse } from "../api/types";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  kind: "normal" | "clarify" | "error";
  tag?: string;
}

interface AssistantPanelProps {
  sessionId: string | null;
  disabled: boolean;
  onPredictionSuccess: (resp: ApiResponse) => void;
}

const CHIPS = [
  "How can I reduce fuel consumption?",
  "Predict my fuel rate",
  "Why is the fuel rate high?",
  "Best way to drive on a grade?",
];

let counter = 0;
const nid = () => `m${++counter}`;

export default function AssistantPanel({
  sessionId,
  disabled,
  onPredictionSuccess,
}: AssistantPanelProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: nid(),
      role: "assistant",
      text:
        "Hi. I can help with synthetic fuel-rate predictions, model explanations, " +
        "and knowledge questions about driving. Ask me in English, Hindi or Hinglish.",
      kind: "normal",
    },
  ]);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  async function send(message?: string) {
    const q = (message ?? text).trim();
    if (!q || !sessionId || sending) return;

    setMessages((m) => [...m, { id: nid(), role: "user", text: q, kind: "normal" }]);
    setText("");
    setSending(true);

    try {
      const resp = await api.sendMessage(sessionId, q);

      const kind = resp.error ? "error" : resp.driver_intent === "clarifying" ? "clarify" : "normal";
      const tag = resp.error
        ? resp.error.message
        : `${resp.driver_intent} · ${resp.delivered ? "delivered" : "not delivered"}`;

      setMessages((m) => [
        ...m,
        {
          id: nid(),
          role: "assistant",
          text: resp.driver_feedback || resp.clarification_request || "",
          kind,
          tag,
        },
      ]);

      const tr = resp.technical_report;
      const fuel =
        tr?.prediction?.predicted_fuel_rate_lph ??
        tr?.explanation?.predicted_fuel_rate_lph ??
        null;
      if (resp.delivered && fuel != null) {
        onPredictionSuccess(resp);
      }
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? `Backend error (${err.status}): ${err.message}`
          : "Could not reach the backend.";
      setMessages((m) => [
        ...m,
        { id: nid(), role: "assistant", text: msg, kind: "error", tag: "request failed" },
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <section className="card area-assistant">
      <div className="card-head">
        <div className="card-title">
          <span className="icon">💬</span> Driver AI assistant
        </div>
        <span className="tag">multi-turn · gateway session</span>
      </div>

      <div className="card-pad">
        <div className="assistant-layout">
          <div className="assistant-thread">
            <div className="assistant-scroll" ref={scrollRef}>
              {messages.map((m) => (
                <div key={m.id} className={`msg ${m.role} ${m.kind === "clarify" ? "clarify" : ""} ${m.kind === "error" ? "error" : ""}`}>
                  <span className="who">{m.role === "user" ? "You" : "Assistant"}</span>
                  <div className="bubble">{m.text}</div>
                  {m.tag && <span className="msg-tag">{m.tag}</span>}
                </div>
              ))}
              {sending && (
                <div className="msg assistant">
                  <span className="who">Assistant</span>
                  <div className="bubble">
                    <span className="typing">
                      <span /><span /><span />
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="assistant-input">
              <textarea
                value={text}
                placeholder="Ask in English / Hindi / Hinglish…"
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                disabled={disabled || sending}
                aria-label="Message the assistant"
              />
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => send()}
                disabled={disabled || sending || !text.trim()}
              >
                Send
              </button>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="card-title" style={{ fontSize: 12 }}>
              Suggested questions
            </div>
            <div className="chips">
              {CHIPS.map((c) => (
                <button
                  key={c}
                  type="button"
                  className="chip"
                  onClick={() => {
                    setText(c);
                    send(c);
                  }}
                  disabled={disabled || sending}
                >
                  {c}
                </button>
              ))}
            </div>

            <div className="advanced-note">
              <span>ℹ️</span>
              <span>
                The assistant keeps a single gateway session across turns, so a
                clarification or prediction can resume with the inputs already
                recorded. Unsupported requests are declined rather than answered
                with invented facts.
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
