document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const statusDiv = document.getElementById("status");
  const errorDiv = document.getElementById("error");
  const progressFill = document.getElementById("progressFill");

  const mTotal = document.getElementById("m_total");
  const mUnique = document.getElementById("m_unique");
  const mAvgLen = document.getElementById("m_avg_len");
  const mDominant = document.getElementById("m_dominant");

  const insightsList = document.getElementById("insightsList");
  const chartContainer = document.getElementById("chart-container");
  const wordcloudAll = document.getElementById("wc_all");
  const wordcloudPos = document.getElementById("wc_positive");
  const wordcloudNeg = document.getElementById("wc_negative");

  const searchInput = document.getElementById("searchInput");
  const sortSelect = document.getElementById("sortSelect");
  const commentsList = document.getElementById("commentsList");
  const commentsCount = document.getElementById("commentsCount");

  const chips = document.querySelectorAll(".chip");

  const YOUTUBE_API_KEY = "AIzaSyAAwUhK2KrdYXj12hfCnErPOZsiGSXtLSo";
  const BACKEND = "http://127.0.0.1:8000";

  let GLOBAL = [];

  /* ---------------- UI helpers ---------------- */
  const setStatus = msg => statusDiv.textContent = msg || "";
  const setError = msg => errorDiv.textContent = msg || "";
  const setProgress = p => progressFill.style.width = `${p}%`;

  function blobToImg(blob, container){
    if(!blob || !container) return;
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      container.innerHTML = "";
      container.appendChild(img);
    };
    img.src = url;
  }

  function normalize(t){ return (t||"").replace(/\s+/g," ").trim(); }

  /* ---------------- Tabs ---------------- */
  document.querySelectorAll(".tab").forEach(tab=>{
    tab.addEventListener("click", ()=>{
      document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.tab).classList.add("active");
    });
  });

  /* ---------------- Comments filtering ---------------- */
  function renderComments(arr){
    commentsList.innerHTML = arr.slice(0,100).map((r,i)=>`
      <div class="comment-card">
        <b>${i+1}.</b> ${r.comment}
        <div class="badge ${r.sentiment}">${r.sentiment}</div>
        <div class="small-note">👍 ${r.likes} | 💬 ${r.replies}</div>
      </div>
    `).join("");
    commentsCount.textContent = `Showing ${Math.min(100,arr.length)} of ${arr.length}`;
  }

  function applyFilters(){
    let arr = [...GLOBAL];
    const q = normalize(searchInput.value).toLowerCase();
    const activeChip = document.querySelector(".chip.active")?.dataset.filter;

    if(activeChip && activeChip !== "all")
      arr = arr.filter(x=>x.sentiment===activeChip);

    if(q) arr = arr.filter(x=>x.comment.toLowerCase().includes(q));

    if(sortSelect.value==="likes") arr.sort((a,b)=>b.likes-a.likes);
    if(sortSelect.value==="replies") arr.sort((a,b)=>b.replies-a.replies);

    renderComments(arr);
  }

  chips.forEach(c=>c.addEventListener("click",()=>{
    chips.forEach(x=>x.classList.remove("active"));
    c.classList.add("active");
    applyFilters();
  }));
  searchInput.addEventListener("input",applyFilters);
  sortSelect.addEventListener("change",applyFilters);

  /* ---------------- Backend calls ---------------- */
  async function fetchChart(counts){
    const r = await fetch(`${BACKEND}/generate_chart`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({sentiment_counts:counts})
    });
    if(!r.ok) throw new Error("Chart failed");
    return r.blob();
  }

  async function fetchWordcloud(texts){
    const r = await fetch(`${BACKEND}/generate_wordcloud`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({comments:texts})
    });
    if(!r.ok) throw new Error("Wordcloud failed");
    return r.blob();
  }

  async function predict(texts){
    const r = await fetch(`${BACKEND}/predict`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({comments:texts})
    });
    if(!r.ok) throw new Error("Prediction failed");
    return r.json();
  }

  /* ---------------- Main analyze ---------------- */
  analyzeBtn.addEventListener("click", async ()=>{
    try{
      setError(""); setStatus("Fetching comments..."); setProgress(10);

      const tabs = await chrome.tabs.query({active:true,currentWindow:true});
      const url = tabs[0].url;

      if(!url.includes("youtube.com/watch")) throw new Error("Only YouTube supported");

      const vid = new URL(url).searchParams.get("v");
      let comments=[];
      let token="";

      while(comments.length<500){
        const res = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${vid}&maxResults=100&pageToken=${token}&key=${YOUTUBE_API_KEY}`);
        const data = await res.json();
        data.items.forEach(i=>{
          const s=i.snippet.topLevelComment.snippet;
          comments.push({text:s.textOriginal, likes:s.likeCount, replies:i.snippet.totalReplyCount});
        });
        token=data.nextPageToken;
        if(!token) break;
      }

      setStatus("Predicting sentiment..."); setProgress(40);

      const preds = await predict(comments.map(c=>c.text));

      GLOBAL = preds.map((p,i)=>({...p,likes:comments[i].likes,replies:comments[i].replies}));

      const counts={positive:0,neutral:0,negative:0};
      GLOBAL.forEach(x=>counts[x.sentiment]++);

      mTotal.textContent=GLOBAL.length;
      mUnique.textContent=new Set(GLOBAL.map(x=>x.comment)).size;
      mAvgLen.textContent=(GLOBAL.reduce((a,b)=>a+b.comment.split(" ").length,0)/GLOBAL.length).toFixed(1)+" words";
      mDominant.textContent=Object.keys(counts).reduce((a,b)=>counts[a]>counts[b]?a:b);

      insightsList.innerHTML=`<li>Positive ${counts.positive}</li><li>Neutral ${counts.neutral}</li><li>Negative ${counts.negative}</li>`;

      setStatus("Generating visuals..."); setProgress(70);

      blobToImg(await fetchChart(counts),chartContainer);

      const allBlob = await fetchWordcloud(GLOBAL.map(x=>x.comment));
      blobToImg(allBlob,wordcloudAll);

      blobToImg(await fetchWordcloud(GLOBAL.filter(x=>x.sentiment==="positive").map(x=>x.comment)),wordcloudPos);
      blobToImg(await fetchWordcloud(GLOBAL.filter(x=>x.sentiment==="negative").map(x=>x.comment)),wordcloudNeg);

      applyFilters();

      setStatus("Done!"); setProgress(100);
    }catch(e){
      console.error(e);
      setError(e.message);
    }
  });
});
