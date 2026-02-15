document.addEventListener("DOMContentLoaded", () => {

  /* ================= CONFIG ================= */
  
  const BACKEND_API_URL = "https://twitter-sentiment-api-pfim.onrender.com";
  const MAX_COMMENTS = 1500;

  /* ================= ELEMENTS ================= */
  const analyzeBtn = document.getElementById("analyzeBtn");
  const statusEl = document.getElementById("status");
  const errorEl = document.getElementById("error");
  const progressFill = document.getElementById("progressFill");

  const mTotal = document.getElementById("m_total");
  const mUnique = document.getElementById("m_unique");
  const mAvgLen = document.getElementById("m_avg_len");
  const mDominant = document.getElementById("m_dominant");
  const insightsList = document.getElementById("insightsList");

  const chartContainer = document.getElementById("chart-container");

  const wcAll = document.getElementById("wc_all");
  const wcPositive = document.getElementById("wc_positive");
  const wcNegative = document.getElementById("wc_negative");

  const commentsList = document.getElementById("commentsList");
  const commentsCount = document.getElementById("commentsCount");
  const searchInput = document.getElementById("searchInput");
  const sortSelect = document.getElementById("sortSelect");
  const exportBtn = document.getElementById("exportBtn");
  const chips = document.querySelectorAll(".chip");

  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");

  /* ================= TAB SWITCH ================= */
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  /* ================= STATE ================= */
  let GLOBAL = [];

  /* ================= HELPERS ================= */
  const setStatus = t => statusEl.textContent = t || "";
  const setError = t => errorEl.textContent = t || "";
  const setProgress = p => progressFill.style.width = `${p}%`;

  const escapeHTML = s =>
    (s || "").replace(/&/g,"&amp;")
             .replace(/</g,"&lt;")
             .replace(/>/g,"&gt;");

  function blobToImg(blob, el){
    if(!el) return;
    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);
    el.innerHTML = "";
    el.classList.remove("small-note");
    el.appendChild(img);
  }

  const isYT = url => url.includes("youtube.com/watch");
  const isReddit = url => url.includes("reddit.com/r/");

  /* ================= FETCH YOUTUBE ================= */
async function fetchYouTube(videoId){
  try {
    const r = await fetch(
      `${BACKEND_API_URL}/youtube-comments?video_id=${videoId}`
    );

    if (!r.ok) {
      throw new Error("Failed to fetch YouTube comments from backend");
    }

    const data = await r.json();

    if (!Array.isArray(data)) {
      throw new Error(data.error || "Invalid response from backend");
    }

    return data.slice(0, MAX_COMMENTS);

  } catch (err) {
    console.error("YouTube fetch error:", err);
    throw err;
  }
}


  /* ================= FETCH REDDIT ================= */
  async function fetchReddit(url){
    const r = await fetch(url.replace(/\/$/,"") + ".json");
    const d = await r.json();
    const res=[];

    const walk = arr => arr.forEach(x=>{
      if(x.data?.body){
        res.push({
          text:x.data.body,
          author:x.data.author,
          likes:x.data.score||0,
          replies:x.data.replies?.data?.children?.length||0
        });
      }
      if(x.data?.replies?.data?.children)
        walk(x.data.replies.data.children);
    });

    walk(d[1]?.data?.children || []);
    return res.slice(0,MAX_COMMENTS);
  }

  /* ================= BACKEND ================= */
  async function predict(texts){
    const r = await fetch(`${BACKEND_API_URL}/predict`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({comments:texts})
    });

    const data = await r.json().catch(()=>null);
    if(!data) throw new Error("Prediction returned no JSON");

    if(Array.isArray(data)) return data;
    if(Array.isArray(data.predictions)) return data.predictions;

    const arr = Object.values(data).find(v => Array.isArray(v));
    if(arr) return arr;

    throw new Error("Invalid prediction response");
  }

  async function pie(counts){
    const r = await fetch(`${BACKEND_API_URL}/generate_chart`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({sentiment_counts:counts})
    });
    return await r.blob();
  }

  async function wordcloud(texts){
    const r = await fetch(`${BACKEND_API_URL}/generate_wordcloud`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({comments:texts})
    });
    return await r.blob();
  }

  /* ================= COMMENTS ================= */
  function renderComments(arr){
    commentsList.innerHTML = arr.slice(0,50).map((x,i)=>`
      <div class="comment-card">
        <b>${i+1}.</b> ${escapeHTML(x.comment)}
        <div class="badge ${x.sentiment}">${x.sentiment}</div>
        <div class="small-note">👍 ${x.likes} | 💬 ${x.replies}</div>
      </div>
    `).join("");

    commentsCount.textContent = `Showing ${Math.min(50,arr.length)} of ${arr.length}`;
  }

  function applyFilters(){
    let a=[...GLOBAL];
    const q=searchInput.value.toLowerCase();
    const f=document.querySelector(".chip.active").dataset.filter;

    if(f!=="all") a=a.filter(x=>x.sentiment===f);
    if(q) a=a.filter(x=>x.comment.toLowerCase().includes(q));

    if(sortSelect.value==="likes") a.sort((x,y)=>y.likes-x.likes);
    if(sortSelect.value==="replies") a.sort((x,y)=>y.replies-x.replies);

    renderComments(a);
  }

  chips.forEach(c=>c.onclick=()=>{
    chips.forEach(x=>x.classList.remove("active"));
    c.classList.add("active");
    applyFilters();
  });

  searchInput.oninput=applyFilters;
  sortSelect.onchange=applyFilters;

  /* ================= MAIN ================= */
  analyzeBtn.onclick = async ()=>{
    try{
      setError("");
      setStatus("Detecting page...");
      setProgress(10);

      const [tab]=await chrome.tabs.query({active:true,currentWindow:true});
      const url=tab.url;

      let raw=[];

      if(isYT(url)){
        setStatus("Fetching YouTube comments...");
        raw=await fetchYouTube(new URL(url).searchParams.get("v"));
      }
      else if(isReddit(url)){
        setStatus("Fetching Reddit comments...");
        raw=await fetchReddit(url);
      }
      else{
        throw new Error("Only YouTube & Reddit supported");
      }

      if(!raw.length) throw new Error("No comments found");

      setProgress(40);

      const texts=raw.map(x=>x.text);
      const preds=await predict(texts);

      GLOBAL = preds.map((p,i)=>({
        comment:texts[i],
        sentiment:p.sentiment || p,
        likes:raw[i]?.likes||0,
        replies:raw[i]?.replies||0
      }));

      setProgress(70);

      const counts={positive:0,neutral:0,negative:0};
      GLOBAL.forEach(x=>{
        if(counts[x.sentiment]!==undefined) counts[x.sentiment]++;
      });

      const total = GLOBAL.length;
      const avgLen = total
        ? (texts.join(" ").split(/\s+/).length / total).toFixed(2)
        : 0;

      mTotal.textContent = total;
      mUnique.textContent = new Set(raw.map(x=>x.author)).size;
      mAvgLen.textContent = avgLen + " words";
      mDominant.textContent =
        Object.entries(counts).sort((a,b)=>b[1]-a[1])[0]?.[0] || "-";

      insightsList.innerHTML = `
        <li>Positive: ${counts.positive}</li>
        <li>Neutral: ${counts.neutral}</li>
        <li>Negative: ${counts.negative}</li>
      `;

      blobToImg(await pie(counts), chartContainer);

      blobToImg(await wordcloud(texts), wcAll);

      if(counts.positive > 0){
        const posTexts = GLOBAL.filter(x=>x.sentiment==="positive").map(x=>x.comment);
        blobToImg(await wordcloud(posTexts), wcPositive);
      }else{
        wcPositive.innerHTML="No positive comments";
      }

      if(counts.negative > 0){
        const negTexts = GLOBAL.filter(x=>x.sentiment==="negative").map(x=>x.comment);
        blobToImg(await wordcloud(negTexts), wcNegative);
      }else{
        wcNegative.innerHTML="No negative comments";
      }

      applyFilters();
      setStatus("Done!");
      setProgress(100);

    }catch(e){
      console.error(e);
      setError(e.message);
      setProgress(0);
    }
  };

});
