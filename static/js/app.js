document.addEventListener("DOMContentLoaded", () => {
    // 1. Navigation
    const navs = [
        { btn: "nav-dashboard", view: "dashboard-view", disp: "grid" },
        { btn: "nav-stats", view: "stats-view", disp: "block" },
        { btn: "nav-history", view: "history-view", disp: "block" },
        { btn: "nav-guides", view: "guides-view", disp: "block" },
        { btn: "nav-about", view: "about-view", disp: "block" }
    ];

    navs.forEach(item => {
        const btn = document.getElementById(item.btn);
        if (btn) {
            btn.addEventListener("click", (e) => {
                e.preventDefault();
                navs.forEach(i => {
                    document.getElementById(i.view).style.display = "none";
                    document.getElementById(i.btn).parentElement.classList.remove("active");
                });
                document.getElementById(item.view).style.display = item.disp;
                btn.parentElement.classList.add("active");
                if (item.btn === "nav-guides") loadGuides();
            });
        }
    });

    // 2. Theme Toggle (Fix)
    const themeToggle = document.getElementById("theme-toggle");
    themeToggle.addEventListener("click", () => {
        document.body.classList.toggle("light-mode");
        const isLight = document.body.classList.contains("light-mode");
        document.getElementById("theme-icon").textContent = isLight ? "🌙" : "☀️";
    });

    // 3. Image Upload & "Change Image" (Fix)
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const changeBtn = document.getElementById("change-btn");
    const preview = document.getElementById("image-preview");
    const analyzeBtn = document.getElementById("analyze-btn");

    const updatePreview = (file) => {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = "block";
            document.querySelector(".upload-placeholder").style.display = "none";
            analyzeBtn.disabled = false;
            changeBtn.style.display = "block";
        };
        reader.readAsDataURL(file);
    };

    browseBtn.addEventListener("click", () => fileInput.click());
    changeBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => updatePreview(e.target.files[0]));

    // 4. Load Encyclopedia
    async function loadGuides() {
        const container = document.getElementById("guides-container");
        if (container.children.length > 0) return;
        const res = await fetch("/get_guides");
        const data = await res.json();
        container.innerHTML = Object.entries(data).map(([key, info]) => `
            <div class="visual-card glass-panel">
                <h4>${key.replace("___", " - ").replace(/_/g, " ")}</h4>
                <p class="subtitle">Biological Care Guide</p>
                <p style="font-size:0.8rem; color:var(--text-muted);">${info.description}</p>
                <p style="margin-top:10px; font-size:0.8rem; color:var(--primary);"><strong>Care:</strong> ${info.treatment}</p>
            </div>
        `).join('');
    }

    // 5. Predict
    analyzeBtn.addEventListener("click", async () => {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        document.getElementById("loading").style.display = "block";
        document.getElementById("result-content").style.display = "none";

        try {
            const res = await fetch("/predict", { method: "POST", body: formData });
            const data = await res.json();
            document.getElementById("loading").style.display = "none";
            if(data.error) { alert(data.error); return; }

            document.getElementById("result-content").style.display = "block";
            document.getElementById("disease-name").textContent = data.disease;
            document.getElementById("confidence-val").textContent = data.confidence;
            document.getElementById("disease-desc").textContent = data.description;
            document.getElementById("disease-treatment").textContent = data.treatment;
        } catch (e) { alert("Server error"); }
    });
});