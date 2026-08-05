import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BookOpenText,
  Brain,
  CalendarCheck,
  CheckCircle2,
  ClipboardList,
  DatabaseZap,
  Filter,
  Search,
  Send,
  Sparkles,
  Upload,
  Utensils,
} from "lucide-react";
import {
  analyzeFoodImage,
  analyzeNutrition,
  askCookHero,
  ChatResponse,
  createDietPlan,
  createMealCheckin,
  DietPlanResponse,
  listTools,
  NutritionAnalysisResponse,
  ToolCatalogItem,
  uploadPersonalDocument,
  VisionAnalyzeResponse,
} from "./api/client";

const examples = ["番茄炒蛋怎么做", "推荐简单的汤", "有鸡蛋和西红柿能吃什么"];
const demoUserId = "demo-user";
const fallbackToolOptions: ToolCatalogItem[] = [
  {
    name: "knowledge_base_search",
    description: "Knowledge search",
    provider: "local",
    input_schema: {},
  },
  { name: "diet_plan", description: "Diet planning", provider: "local", input_schema: {} },
  { name: "meal_checkin", description: "Meal check-in", provider: "local", input_schema: {} },
  { name: "nutrition_analysis", description: "Nutrition analysis", provider: "local", input_schema: {} },
  {
    name: "diet_planning_expert",
    description: "Diet planning expert",
    provider: "subagent",
    input_schema: {},
  },
  {
    name: "meal_record_expert",
    description: "Meal record expert",
    provider: "subagent",
    input_schema: {},
  },
  {
    name: "nutrition_analysis_expert",
    description: "Nutrition analysis expert",
    provider: "subagent",
    input_schema: {},
  },
];

function toolLabel(tool: ToolCatalogItem) {
  return `${tool.name.replace(/_/g, " ")} · ${tool.provider}`;
}

export function App() {
  const [message, setMessage] = useState(examples[0]);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [personalTitle, setPersonalTitle] = useState("训练日晚餐偏好");
  const [personalContent, setPersonalContent] = useState("# 训练日晚餐偏好\n\n偏好高蛋白、低油，避免太辣。");
  const [personalStatus, setPersonalStatus] = useState<string>("Not indexed");
  const [imageUrl, setImageUrl] = useState("");
  const [visionResult, setVisionResult] = useState<VisionAnalyzeResponse | null>(null);
  const [dietGoal, setDietGoal] = useState("减脂高蛋白，晚餐少油");
  const [dietDays, setDietDays] = useState(7);
  const [dietContext, setDietContext] = useState("番茄炒蛋和简单汤品");
  const [dietPlan, setDietPlan] = useState<DietPlanResponse | null>(null);
  const [checkinMealTime, setCheckinMealTime] = useState("dinner");
  const [checkinDescription, setCheckinDescription] = useState("番茄炒蛋、米饭和一份青菜");
  const [checkinStatus, setCheckinStatus] = useState("No check-in yet");
  const [nutritionDate, setNutritionDate] = useState(new Date().toISOString().slice(0, 10));
  const [nutritionResult, setNutritionResult] = useState<NutritionAnalysisResponse | null>(null);
  const [selectedTools, setSelectedTools] = useState<string[]>(["knowledge_base_search"]);
  const [availableTools, setAvailableTools] = useState<ToolCatalogItem[]>(fallbackToolOptions);

  useEffect(() => {
    let cancelled = false;
    listTools()
      .then((tools) => {
        if (!cancelled && tools.length) {
          setAvailableTools(tools);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setAvailableTools(fallbackToolOptions);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    if (!message.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await askCookHero(message.trim(), demoUserId, selectedTools));
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败");
    } finally {
      setLoading(false);
    }
  }


  function toggleTool(name: string) {
    setSelectedTools((current) => {
      if (current.includes(name)) {
        return current.length > 1 ? current.filter((item) => item !== name) : current;
      }
      return [...current, name];
    });
  }

  async function submitPersonal(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const payload = await uploadPersonalDocument({
        user_id: demoUserId,
        title: personalTitle,
        content: personalContent,
        category: "个人饮食",
        difficulty: "普通",
      });
      setPersonalStatus(payload.indexed ? `Indexed ${payload.doc_id.slice(0, 8)}` : "Failed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "个人知识入库失败");
    }
  }

  async function submitVision(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      setVisionResult(await analyzeFoodImage({ image_url: imageUrl, user_goal: dietGoal }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "图片分析失败");
    }
  }

  async function submitDietPlan(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      setDietPlan(
        await createDietPlan({
          user_id: demoUserId,
          goal: dietGoal,
          days: dietDays,
          context_query: dietContext,
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "饮食计划生成失败");
    }
  }

  async function submitCheckin(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const checkin = await createMealCheckin({
        user_id: demoUserId,
        meal_time: checkinMealTime,
        description: checkinDescription,
        image_url: imageUrl || undefined,
        user_goal: dietGoal || undefined,
        image_analysis: visionResult ?? undefined,
      });
      setCheckinStatus(`Saved ${checkin.checkin_id.slice(0, 8)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "打卡保存失败");
    }
  }

  async function submitNutrition(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      setNutritionResult(await analyzeNutrition({ user_id: demoUserId, date: nutritionDate }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "营养分析失败");
    }
  }

  const trace = useMemo(() => result?.trace ?? [], [result]);
  const nutritionRows = Object.entries(
    nutritionResult?.metrics ?? { protein: "--", carbs: "--", fat: "--", energy: "--", risk: "--" }
  );

  return (
    <main className="shell">
      <aside className="rail" aria-label="CookHero navigation">
        <div className="brand-mark">
          <Utensils size={20} />
        </div>
        <button className="rail-button is-active" aria-label="智能问答">
          <Brain size={18} />
        </button>
        <button className="rail-button" aria-label="知识库">
          <DatabaseZap size={18} />
        </button>
        <button className="rail-button" aria-label="营养看板">
          <Activity size={18} />
        </button>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Multi-Agent Diet OS</p>
            <h1>CookHero</h1>
          </div>
          <div className="status-pill">
            <CheckCircle2 size={16} />
            RAG Ready
          </div>
        </header>

        <section className="query-band">
          <div className="query-copy">
            <h2>饮食计划、打卡记录和菜谱检索在同一个工作台完成。</h2>
            <p>输入食材、菜名或目标，系统会先做查询改写，再走元数据过滤、混合检索、重排序和父文档还原。</p>
          </div>
          <form className="ask-box" onSubmit={submit}>
            <label htmlFor="message">Ask</label>
            <div className="input-row">
              <Search size={18} />
              <input
                id="message"
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="例如：推荐简单的汤"
              />
              <button type="submit" disabled={loading} aria-label="发送">
                {loading ? <Sparkles size={18} /> : <Send size={18} />}
              </button>
            </div>
            <div className="quick-row">
              {examples.map((item) => (
                <button key={item} type="button" onClick={() => setMessage(item)}>
                  {item}
                </button>
              ))}
            </div>
            <div className="tool-scope" aria-label="Enabled agent tools">
              {availableTools.map((tool) => (
                <label key={tool.name} title={tool.description}>
                  <input
                    type="checkbox"
                    checked={selectedTools.includes(tool.name)}
                    onChange={() => toggleTool(tool.name)}
                  />
                  <span>{toolLabel(tool)}</span>
                </label>
              ))}
            </div>
          </form>
        </section>

        {error && <div className="error-line">{error}</div>}

        <section className="content-grid">
          <article className="answer-panel">
            <div className="panel-head">
              <BookOpenText size={18} />
              <h2>Agent Response</h2>
            </div>
            <p className="answer-text">
              {result?.answer ?? "等待问题输入。返回后这里会展示智能体整理后的饮食建议。"}
            </p>
            <div className="context-preview">
              <span>Context</span>
              <pre>{result?.context_preview || "No retrieval context yet."}</pre>
            </div>
          </article>

          <article className="trace-panel">
            <div className="panel-head">
              <Filter size={18} />
              <h2>Agent Trace</h2>
            </div>
            <dl className="facts">
              <div>
                <dt>Thought</dt>
                <dd>{result?.thought ?? "-"}</dd>
              </div>
              <div>
                <dt>Action</dt>
                <dd>{result?.action ?? "-"}</dd>
              </div>
              <div>
                <dt>Rewrite</dt>
                <dd>{result?.rewritten_query ?? "-"}</dd>
              </div>
              <div>
                <dt>Filter</dt>
                <dd>{result?.metadata_expression ?? "NONE"}</dd>
              </div>
            </dl>
            <ol className="trace-list">
              {(trace.length ? trace : ["rewrite", "metadata", "cache", "hybrid", "rerank", "parent"]).map(
                (item, index) => (
                  <li key={`${item}-${index}`}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    {item}
                  </li>
                )
              )}
            </ol>
          </article>

          <article className="sources-panel">
            <div className="panel-head">
              <DatabaseZap size={18} />
              <h2>Sources</h2>
            </div>
            <div className="source-list">
              {(result?.sources ?? []).map((source) => (
                <div className="source-row" key={`${source.source}-${source.score}`}>
                  <div>
                    <strong>{source.title}</strong>
                    <span>{source.category} · {source.difficulty}</span>
                  </div>
                  <code>{source.score?.toFixed(3) ?? "-"}</code>
                </div>
              ))}
              {!result?.sources?.length && <p className="muted">检索后会展示命中的父文档来源。</p>}
            </div>
          </article>

          <article className="sources-panel">
            <div className="panel-head">
              <ClipboardList size={18} />
              <h2>Diet Plan</h2>
            </div>
            <form className="stack-form" onSubmit={submitDietPlan}>
              <textarea value={dietGoal} onChange={(event) => setDietGoal(event.target.value)} />
              <div className="inline-row">
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={dietDays}
                  onChange={(event) => setDietDays(Number(event.target.value))}
                  aria-label="天数"
                />
                <input value={dietContext} onChange={(event) => setDietContext(event.target.value)} />
              </div>
              <button type="submit">Generate</button>
            </form>
            {dietPlan && (
              <div className="result-block">
                <strong>{dietPlan.goal}</strong>
                <pre>{dietPlan.content}</pre>
              </div>
            )}
          </article>

          <article className="sources-panel">
            <div className="panel-head">
              <Upload size={18} />
              <h2>Knowledge Intake</h2>
            </div>
            <form className="stack-form" onSubmit={submitPersonal}>
              <input value={personalTitle} onChange={(event) => setPersonalTitle(event.target.value)} />
              <textarea value={personalContent} onChange={(event) => setPersonalContent(event.target.value)} />
              <button type="submit">Index</button>
            </form>
            <p className="muted">{personalStatus}</p>
          </article>

          <article className="nutrition-panel">
            <div className="panel-head">
              <Sparkles size={18} />
              <h2>Vision Analysis</h2>
            </div>
            <form className="stack-form" onSubmit={submitVision}>
              <input value={imageUrl} onChange={(event) => setImageUrl(event.target.value)} placeholder="Image URL" />
              <button type="submit">Analyze</button>
            </form>
            {visionResult && (
              <div className="vision-result">
                <strong>{visionResult.dish_name}</strong>
                <span>{visionResult.ingredients.join(" · ")}</span>
              </div>
            )}
          </article>

          <article className="nutrition-panel">
            <div className="panel-head">
              <CalendarCheck size={18} />
              <h2>Meal Check-in</h2>
            </div>
            <form className="stack-form" onSubmit={submitCheckin}>
              <div className="inline-row">
                <input value={checkinMealTime} onChange={(event) => setCheckinMealTime(event.target.value)} />
                <input value={nutritionDate} onChange={(event) => setNutritionDate(event.target.value)} />
              </div>
              <textarea value={checkinDescription} onChange={(event) => setCheckinDescription(event.target.value)} />
              <button type="submit">Save</button>
            </form>
            <p className="muted">{checkinStatus}</p>
          </article>

          <article className="nutrition-panel">
            <div className="panel-head">
              <Activity size={18} />
              <h2>Nutrition Board</h2>
            </div>
            <form className="stack-form" onSubmit={submitNutrition}>
              <input value={nutritionDate} onChange={(event) => setNutritionDate(event.target.value)} />
              <button type="submit">Analyze</button>
            </form>
            <div className="metric-grid">
              {nutritionRows.map(([name, value]) => (
                <div key={name}>
                  <span>{name}</span>
                  <strong>{String(value)}</strong>
                </div>
              ))}
            </div>
            {nutritionResult && <p className="answer-text compact">{nutritionResult.content}</p>}
          </article>
        </section>
      </section>
    </main>
  );
}
