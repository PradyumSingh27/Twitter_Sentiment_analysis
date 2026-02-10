document.addEventListener("DOMContentLoaded", () => {

  /* ================== CONFIG ================== */
  const YOUTUBE_API_KEY = "AIzaSyAAwUhK2KrdYXj12hfCnErPOZsiGSXtLSo";
  const BACKEND_API_URL = "http://127.0.0.1:8000";

  const RAPIDAPI_KEY = "1b0959985cmsh66fd3d2dac74ee3p10be48jsn16cb3bf73db9";
  const RAPIDAPI_HOST = "twitter-api45.p.rapidapi.com";

  const MAX_COMMENTS = 500;

  /* ================== ELEMENTS ================== */
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

  /* ================= TAB SWITCHING ================= */
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      // remove active from all
      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      // activate clicked tab
      tab.classList.add("active");
      const targetId = tab.dataset.tab;
      const targetPanel = document.getElementById(targetId);

      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });


  /* ================== STATE ================== */
  let GLOBAL = [];

  /* ================== HELPERS ================== */
  const setStatus = t => statusEl.textContent = t || "";
  const setError = t => errorEl.textContent = t || "";
  const setProgress = p => progressFill.style.width = `${p}%`;

  const escapeHTML = s =>
    s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  function blobToImg(blob, el){
    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);
    el.innerHTML = "";
    el.classList.remove("small-note");
    el.appendChild(img);
  }

  /* ================== URL DETECT ================== */
  const isYT = url => url.includes("youtube.com/watch");
  const isReddit = url => url.includes("reddit.com/r/");
  const isTwitter = url =>
    url.includes("twitter.com") || url.includes("x.com");

  /* ================== FETCH YOUTUBE ================== */
  async function fetchYouTube(videoId){
    let out = [], page = "";
    while(out.length < MAX_COMMENTS){
      const url =
        `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&key=${YOUTUBE_API_KEY}` +
        (page ? `&pageToken=${page}` : "");
      const r = await fetch(url);
      const d = await r.json();
      if(!d.items) break;

      d.items.forEach(i=>{
        const s=i.snippet.topLevelComment.snippet;
        out.push({
          text:s.textOriginal,
          author:s.authorChannelId?.value || "yt",
          likes:s.likeCount||0,
          replies:i.snippet.totalReplyCount||0
        });
      });
      page=d.nextPageToken;
      if(!page) break;
    }
    return out;
  }

  /* ================== FETCH REDDIT ================== */
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
    walk(d[1].data.children);
    return res.slice(0,MAX_COMMENTS);
  }

  /* ================== FETCH TWITTER ================== */
  async function fetchTwitter(url){
    const id = url.split("/").pop();
    const r = await fetch(
      `https://${RAPIDAPI_HOST}/tweet/thread.php?id=${id}`,
      { headers:{
        "x-rapidapi-key":RAPIDAPI_KEY,
        "x-rapidapi-host":RAPIDAPI_HOST
      }}
    );
    const d = await r.json();
    return d.timeline.map(t=>({
      text:t.text,
      author:t.screen_name,
      likes:t.favorites||0,
      replies:t.replies||0
    })).slice(0,MAX_COMMENTS);
  }

  /* ================== BACKEND ================== */
  async function predict(texts){
    const r = await fetch(`${BACKEND_API_URL}/predict`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({comments:texts})
    });
    return await r.json();
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

  /* ================== COMMENTS UI ================== */
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

  /* ================== MAIN ================== */
  analyzeBtn.onclick = async ()=>{
    try{
      setError(""); setStatus("Detecting page...");
      setProgress(10);

      const [tab]=await chrome.tabs.query({active:true,currentWindow:true});
      const url=tab.url;
      let raw=[];

      if(isYT(url)){
        setStatus("Fetching YouTube comments...");
        raw=await fetchYouTube(new URL(url).searchParams.get("v"));
      }else if(isReddit(url)){
        setStatus("Fetching Reddit comments...");
        raw=await fetchReddit(url);
      }else if(isTwitter(url)){
        setStatus("Fetching Twitter tweets...");
        raw=await fetchTwitter(url);
      }else{
        throw new Error("Unsupported site");
      }

      setProgress(40);
      const texts=raw.map(x=>x.text);
      const preds=await predict(texts);

      GLOBAL = preds.map((p,i)=>({
        ...p,
        likes:raw[i].likes,
        replies:raw[i].replies
      }));

      setProgress(70);

      const counts={positive:0,neutral:0,negative:0};
      GLOBAL.forEach(x=>counts[x.sentiment]++);

      mTotal.textContent=GLOBAL.length;
      mUnique.textContent=new Set(raw.map(x=>x.author)).size;
      mAvgLen.textContent=(texts.join(" ").split(/\s+/).length/GLOBAL.length).toFixed(2)+" words";
      mDominant.textContent=Object.entries(counts).sort((a,b)=>b[1]-a[1])[0][0];

      insightsList.innerHTML=`
        <li>Positive: ${counts.positive}</li>
        <li>Neutral: ${counts.neutral}</li>
        <li>Negative: ${counts.negative}</li>
      `;

      blobToImg(await pie(counts), chartContainer);
      blobToImg(await wordcloud(texts), wcAll);
      blobToImg(await wordcloud(GLOBAL.filter(x=>x.sentiment==="positive").map(x=>x.comment)), wcPositive);
      blobToImg(await wordcloud(GLOBAL.filter(x=>x.sentiment==="negative").map(x=>x.comment)), wcNegative);

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
