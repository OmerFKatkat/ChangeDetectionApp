/* ═══════════════════════════════════════════════════════════════════════════
   Satellite Change Detection — Frontend Logic
   Two ways to pick an AOI:
     • Browse: cascading İl → İlçe → Mahalle dropdowns, then click an AOI card.
     • Coords: type lat/lon, the backend resolves to the AOI whose bbox covers
       the point (or returns "no_coverage" for an inline info banner).
   All curated AOIs are fetched from /api/aois at boot and drawn on the map
   as low-opacity rectangles + markers. Selection highlights both. Inference
   uses the AOI id (or falls back to lat/lon on the server).
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
    'use strict';

    const $  = (sel)        => document.querySelector(sel);
    const $$ = (sel, root)  => Array.from((root || document).querySelectorAll(sel));

    // ── DOM ────────────────────────────────────────────────────────────────────

    // Tabs
    const tabBrowse   = $('#tab-browse');
    const tabCoords   = $('#tab-coords');
    const panelBrowse = $('#panel-browse');
    const panelCoords = $('#panel-coords');

    // Browse panel
    const selectIl       = $('#select-il');
    const selectIlce     = $('#select-ilce');
    const selectMahalle  = $('#select-mahalle');
    const fieldMahalle   = $('#field-mahalle');
    const aoiListEl      = $('#aoi-list');
    const aoiListEmptyEl = $('#aoi-list-empty');
    const aoiCountEl     = $('#aoi-count');

    // Coords panel
    const inputLat       = $('#input-lat');
    const inputLon       = $('#input-lon');
    const btnFind        = $('#btn-find');
    const coordInfo      = $('#coord-info');
    const coordAoiListEl = $('#coord-aoi-list');

    // Advanced + action
    const btnDetect      = $('#btn-detect');
    const btnText        = $('.btn-text');
    const btnIcon        = $('.btn-icon');
    const btnLoader      = $('#btn-loader');
    const btnNew         = $('#btn-new');
    const btnDownload    = $('#btn-download');
    const toast          = $('#error-toast');
    const toastMsg       = $('#error-msg');

    // Results
    const resultsSection = $('#results-section');
    const resultBefore   = $('#result-before');
    const resultAfter    = $('#result-after');
    const resultMask     = $('#result-mask');
    const resultOverlay  = $('#result-overlay');
    const sliderBefore   = $('#slider-before');
    const sliderAfter    = $('#slider-after');
    const statPct        = $('#stat-pct');
    const statConfidence = $('#stat-confidence');
    const statSize       = $('#stat-size');
    const infoArea       = $('#info-area');
    const infoLocation   = $('#info-location');
    const infoPreDate    = $('#info-pre-date');
    const infoPostDate   = $('#info-post-date');

    // Side/slider toggle
    const toggleSide      = $('#toggle-side');
    const toggleSlider    = $('#toggle-slider');
    const sideView        = $('#side-view');
    const sliderView      = $('#slider-view');
    const sliderContainer = $('#slider-container');
    const sliderOverlayEl = $('#slider-overlay');
    const sliderHandle    = $('#slider-handle');

    // ── State ──────────────────────────────────────────────────────────────────

    /** Flat catalog from /api/aois — array of { id, name, il, ilce, mahalle, lat, lon, aoi_m, pre_date, post_date }. */
    let aoisFlat = [];
    /** id → aoi for fast lookup. */
    const aoisById = {};
    /** { il: { ilce: [mahalle | null, ...] } } from /api/aois. */
    let hierarchy = {};
    /** Sentinel used in the mahalle <select> for AOIs whose mahalle is null. */
    const NULL_MAHALLE = '__null__';

    let selectedAoiId = null;
    let resultData    = null;

    // Leaflet layers — per-AOI rectangles and markers stay drawn for the
    // whole session; only the "selected" overlay rect changes on selection.
    /** id → L.rectangle (low-opacity, all-AOIs layer). */
    const aoiRects = {};
    /** id → L.marker. */
    const aoiMarkers = {};
    /** Highlighted rect for the currently selected AOI (sits above aoiRects). */
    let selectedRectLayer = null;

    let queryMarker = null
    // ── Pick Map (Leaflet) ─────────────────────────────────────────────────────

    const ESRI_TILES = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
    const ESRI_ATTR  = 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community';

    const pickMap = L.map('pick-map', {
        center: [38.5, 36.5],   // rough centroid of the curated set (Türkiye)
        zoom: 6,
        zoomControl: true,
    });
    L.tileLayer(ESRI_TILES, { attribution: ESRI_ATTR, maxZoom: 19 }).addTo(pickMap);

    /** Same flat-earth bbox formula the backend uses (custom_aois.compute_bbox).
     *  Returns Leaflet-style [[lat_min, lon_min], [lat_max, lon_max]]. */
    function aoiBoundsLatLon(lat, lon, aoiM) {
        const dLat = (aoiM / 2) / 111320.0;
        const dLon = (aoiM / 2) / (111320.0 * Math.cos(lat * Math.PI / 180));
        return [[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]];
    }

    function fmtCoords(lat, lon) {
        const latDir = lat >= 0 ? 'N' : 'S';
        const lonDir = lon >= 0 ? 'E' : 'W';
        return `${Math.abs(lat).toFixed(4)}°${latDir}, ${Math.abs(lon).toFixed(4)}°${lonDir}`;
    }

    function makeAoiIcon() {
        return L.divIcon({
            className: 'aoi-marker',
            html: `<div class="aoi-marker-pin"></div>`,
            iconSize:   [22, 30],
            iconAnchor: [11, 30],
            popupAnchor: [0, -28],
        });
    }

    function makeQueryIcon(){
        return L.divIcon({
            className: 'query-marker',
            html: '<div class="query-marker-dot"></div>"',
            iconSize: [18,18],
            iconAnchor: [9,9],
        });
    }

    function placeQueryMarker(lat, lon){
        if(queryMarker){
            queryMarker.setLatLng([lat,lon]);
        }else {
            queryMarker = L.marker([lat, lon],{
                icon: makeQueryIcon(),
                interactive: false,
                zIndexOffset: 500,
            }).addTo(pickMap);
        }
    }
    function clearQueryMarker(){
        if (queryMarker){ queryMarker.remove(); queryMarker = null;
        }
    }

    // ── Locale helpers ─────────────────────────────────────────────────────────

    /** Sort strings with the Turkish locale so İ/ı, Ş, Ğ etc. land correctly. */
    function trSort(arr) {
        return [...arr].sort((a, b) => String(a).localeCompare(String(b), 'tr'));
    }

    /** UI label for a possibly-null mahalle. */
    function mahalleLabel(m) {
        return (m == null || m === '') ? '(Genel)' : m;
    }

    /** "Hatay · Antakya · Merkez" — skip mahalle if null. */
    function locationLabel(aoi) {
        const parts = [aoi.il, aoi.ilce];
        if (aoi.mahalle) parts.push(aoi.mahalle);
        return parts.join(' · ');
    }

    // ── Initial fetch ──────────────────────────────────────────────────────────

    async function bootstrap() {
        try {
            const res = await fetch('/api/aois');
            if (!res.ok) throw new Error(`/api/aois → ${res.status}`);
            const data = await res.json();
            aoisFlat = data.aois || [];
            hierarchy = data.hierarchy || {};
            aoisFlat.forEach(a => { aoisById[a.id] = a; });
        } catch (err) {
            showError(`Could not load AOI catalog: ${err.message}`);
            return;
        }
        renderMapLayers();
        fitMapToAllAOIs();
        populateIlDropdown();
    }

    // ── Map: draw all AOIs ─────────────────────────────────────────────────────

    function renderMapLayers() {
        aoisFlat.forEach(aoi => {
            const bounds = aoiBoundsLatLon(aoi.lat, aoi.lon, aoi.aoi_m);
            const rect = L.rectangle(bounds, {
                color: '#638cff',
                weight: 1,
                fillColor: '#638cff',
                fillOpacity: 0.06,
                opacity: 0.45,
                interactive: false,
            }).addTo(pickMap);
            aoiRects[aoi.id] = rect;

            const marker = L.marker([aoi.lat, aoi.lon], {
                icon: makeAoiIcon(),
                title: aoi.name,
                riseOnHover: true,
            }).addTo(pickMap);
            marker.bindTooltip(aoi.name, {
                direction: 'top',
                offset: [0, -28],
                opacity: 0.95,
            });
            // Clicking a marker is equivalent to clicking its card.
            marker.on('click', () => selectAoi(aoi.id, { fromMap: true }));
            aoiMarkers[aoi.id] = marker;
        });
    }

    function fitMapToAllAOIs() {
        if (!aoisFlat.length) return;
        const bounds = L.latLngBounds([]);
        aoisFlat.forEach(a => bounds.extend([a.lat, a.lon]));
        if (bounds.isValid()) {
            pickMap.fitBounds(bounds, { padding: [60, 60], maxZoom: 11 });
        }
    }

    // ── Browse tab: cascading dropdowns ────────────────────────────────────────

    function populateIlDropdown() {
        const iller = trSort(Object.keys(hierarchy));
        selectIl.innerHTML = '<option value="">Seçiniz…</option>'
            + iller.map(il => `<option value="${escapeHtml(il)}">${escapeHtml(il)}</option>`).join('');
        selectIl.disabled = iller.length === 0;
        resetIlce();
    }

    function resetIlce() {
        selectIlce.innerHTML = '<option value="">—</option>';
        selectIlce.disabled = true;
        resetMahalle();
    }

    function resetMahalle() {
        selectMahalle.innerHTML = '<option value="">—</option>';
        selectMahalle.disabled = true;
        fieldMahalle.style.visibility = 'hidden';   // keep layout space, hide visually
    }

    function onIlChange() {
        const il = selectIl.value;
        if (!il || !hierarchy[il]) {
            resetIlce();
            renderBrowseAois([]);
            return;
        }
        const ilceler = trSort(Object.keys(hierarchy[il]));
        selectIlce.innerHTML = '<option value="">Seçiniz…</option>'
            + ilceler.map(i => `<option value="${escapeHtml(i)}">${escapeHtml(i)}</option>`).join('');
        selectIlce.disabled = false;
        resetMahalle();
        renderBrowseAois([]);
    }

    function onIlceChange() {
        const il   = selectIl.value;
        const ilce = selectIlce.value;
        if (!il || !ilce || !hierarchy[il] || !hierarchy[il][ilce]) {
            resetMahalle();
            renderBrowseAois([]);
            return;
        }
        const mahalleler = hierarchy[il][ilce] || [];

        // If the only mahalle bucket is null, there's nothing to disambiguate —
        // just render every AOI under this il/ilçe.
        const onlyNull = mahalleler.length > 0 && mahalleler.every(m => m == null);
        if (mahalleler.length === 0 || onlyNull) {
            resetMahalle();
            renderBrowseAois(filterAois({ il, ilce }));
            return;
        }

        // Otherwise show the mahalle dropdown; null becomes a "(Genel)" entry.
        fieldMahalle.style.visibility = 'visible';
        selectMahalle.disabled = false;
        // Sort with null treated as "(Genel)" string for stable ordering.
        const labeled = mahalleler.map(m => ({
            raw: m,
            value: m == null ? NULL_MAHALLE : m,
            label: mahalleLabel(m),
        }));
        labeled.sort((a, b) => a.label.localeCompare(b.label, 'tr'));
        selectMahalle.innerHTML = '<option value="">Seçiniz…</option>'
            + labeled.map(o =>
                `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`
            ).join('');
        renderBrowseAois([]);    // wait for user to pick a mahalle
    }

    function onMahalleChange() {
        const il   = selectIl.value;
        const ilce = selectIlce.value;
        const mraw = selectMahalle.value;
        if (!il || !ilce || !mraw) {
            renderBrowseAois([]);
            return;
        }
        const mahalle = mraw === NULL_MAHALLE ? null : mraw;
        renderBrowseAois(filterAois({ il, ilce, mahalle }));
    }

    function filterAois({ il, ilce, mahalle }) {
        return aoisFlat.filter(a => {
            if (il   !== undefined && a.il   !== il)   return false;
            if (ilce !== undefined && a.ilce !== ilce) return false;
            // `undefined` = don't filter by mahalle; explicit `null` = match only null.
            if (mahalle !== undefined && a.mahalle !== mahalle) return false;
            return true;
        });
    }

    // ── Card rendering ─────────────────────────────────────────────────────────

    function renderBrowseAois(aois) {
        renderCardsInto(aoiListEl, aois);
        const n = aois.length;
        aoiCountEl.textContent = n === 0 ? '—'
            : `${n} area${n === 1 ? '' : 's'}`;
        if (aoiListEmptyEl) {
            aoiListEmptyEl.style.display = n === 0 ? 'block' : 'none';
        }
    }

    function renderCoordResult(aoi) {
        renderCardsInto(coordAoiListEl, aoi ? [aoi] : []);
    }

    function renderCardsInto(container, aois) {
        // Preserve the empty-state node (it only lives in the browse list) so we
        // don't lose it on re-render.
        const emptyEl = container.querySelector('.aoi-list-empty');
        container.innerHTML = '';
        if (emptyEl) container.appendChild(emptyEl);

        aois.forEach(aoi => {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = 'aoi-card';
            card.dataset.aoiId = aoi.id;
            card.setAttribute('role', 'option');
            card.setAttribute('aria-selected', aoi.id === selectedAoiId ? 'true' : 'false');
            if (aoi.id === selectedAoiId) card.classList.add('selected');
            card.innerHTML = `
                <div class="aoi-card-name">${escapeHtml(aoi.name)}</div>
                <div class="aoi-card-coords">${fmtCoords(aoi.lat, aoi.lon)}</div>
                <div class="aoi-card-meta">
                    <span class="aoi-card-size">${aoi.aoi_m}m × ${aoi.aoi_m}m</span>
                    <span class="aoi-card-dates">Pre: ${aoi.pre_date} · Post: ${aoi.post_date}</span>
                </div>
            `;
            card.addEventListener('click', () => selectAoi(aoi.id));
            container.appendChild(card);
        });
    }

    // ── Selection ──────────────────────────────────────────────────────────────

    function selectAoi(aoiId, opts = {}) {
        const aoi = aoisById[aoiId];
        if (!aoi) return;
        selectedAoiId = aoiId;

        // Update visual state on EVERY card in EVERY list — selection is
        // global, not per-tab.
        $$('.aoi-card').forEach(card => {
            const isSel = card.dataset.aoiId === aoiId;
            card.classList.toggle('selected', isSel);
            card.setAttribute('aria-selected', isSel ? 'true' : 'false');
            if (isSel && !opts.fromMap) {
                card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }
        });

        // Markers: only one selected at a time.
        Object.entries(aoiMarkers).forEach(([id, m]) => {
            const el = m.getElement();
            if (!el) return;
            el.classList.toggle('selected', id === aoiId);
        });

        // Zoom + draw the highlighted bbox.
        pickMap.setView([aoi.lat, aoi.lon], 17, { animate: true });
        const bounds = aoiBoundsLatLon(aoi.lat, aoi.lon, aoi.aoi_m);
        if (selectedRectLayer) {
            selectedRectLayer.setBounds(bounds);
        } else {
            selectedRectLayer = L.rectangle(bounds, {
                color: '#638cff',
                weight: 2,
                fillColor: '#638cff',
                fillOpacity: 0.10,
                interactive: false,
            }).addTo(pickMap);
        }

        btnDetect.disabled = false;
    }

    function clearSelection() {
        selectedAoiId = null;
        if (selectedRectLayer) { selectedRectLayer.remove(); selectedRectLayer = null; }
        $$('.aoi-card').forEach(card => {
            card.classList.remove('selected');
            card.setAttribute('aria-selected', 'false');
        });
        Object.values(aoiMarkers).forEach(m => {
            const el = m.getElement();
            if (el) el.classList.remove('selected');
        });
        btnDetect.disabled = true;
    }

    // ── Tabs ───────────────────────────────────────────────────────────────────

    function setTab(name) {
        const isBrowse = name === 'browse';
        tabBrowse.classList.toggle('active', isBrowse);
        tabCoords.classList.toggle('active', !isBrowse);
        tabBrowse.setAttribute('aria-selected', isBrowse ? 'true' : 'false');
        tabCoords.setAttribute('aria-selected', !isBrowse ? 'true' : 'false');
        panelBrowse.hidden = !isBrowse;
        panelCoords.hidden =  isBrowse;
    }

    // ── Coordinates tab ────────────────────────────────────────────────────────

    function showCoordInfo(message) {
        coordInfo.textContent = message;
        coordInfo.hidden = false;
    }

    function clearCoordInfo() {
        coordInfo.textContent = '';
        coordInfo.hidden = true;
    }

    async function onFindAoi() {
        clearCoordInfo();
        renderCoordResult(null);

        const lat = parseFloat(inputLat.value);
        const lon = parseFloat(inputLon.value);

        if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
            showCoordInfo('Please enter both latitude and longitude as numbers.');
            return;
        }
        if (lat < -90 || lat > 90) {
            showCoordInfo('Latitude must be between -90 and 90.');
            return;
        }
        if (lon < -180 || lon > 180) {
            showCoordInfo('Longitude must be between -180 and 180.');
            return;
        }

        btnFind.disabled = true;
        placeQueryMarker(lat,lon);
        try {
            const url = `/api/aois/lookup?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`;
            const res = await fetch(url);
            const data = await res.json().catch(() => ({}));

            if (res.status === 422 && data.error === 'no_coverage') {
                // Friendly inline message — not a toast.
                showCoordInfo(data.message ||
                    "Currently we don't have coverage at this location, but we may add more areas in the future.");
                pickMap.setView([lat, lon], 13, { animate: true });
                return;
            }
            if (!res.ok || data.found === false) {
                throw new Error(data.detail || data.message || `Lookup failed (${res.status}).`);
            }

            const aoi = data.aoi;
            if (!aoi || !aoi.id) throw new Error('Malformed lookup response.');

            // Cache it in case the catalog response and lookup ever drift.
            aoisById[aoi.id] = aoisById[aoi.id] || aoi;
            renderCoordResult(aoi);
            selectAoi(aoi.id);
        } catch (err) {
            showError(err.message);
        } finally {
            btnFind.disabled = false;
        }
    }

    // ── Detection ──────────────────────────────────────────────────────────────

    async function runDetection() {
        if (!selectedAoiId) {
            showError('Pick an AOI first.');
            return;
        }

        const body = {
            aoi_id:    selectedAoiId,
            threshold: 0.5,
        };

        btnText.style.display   = 'none';
        btnIcon.style.display   = 'none';
        btnLoader.style.display = 'inline-flex';
        btnDetect.disabled      = true;

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await response.json();
            if (!response.ok || data.success === false) {
                if (response.status === 422 && data.error === 'no_coverage') {
                    throw new Error(data.message ||
                        "Currently we don't have coverage at this location.");
                }
                throw new Error(data.detail || data.error || `Server returned ${response.status}`);
            }
            resultData = data;
            showResults(data);
        } catch (err) {
            showError(err.message);
        } finally {
            btnText.style.display   = 'inline';
            btnIcon.style.display   = 'flex';
            btnLoader.style.display = 'none';
            btnDetect.disabled      = false;
        }
    }

    // ── Results ────────────────────────────────────────────────────────────────

    function b64png(d) { return `data:image/png;base64,${d}`; }

    function showResults(data) {
        resultBefore.src  = b64png(data.before);
        resultAfter.src   = b64png(data.after);
        resultMask.src    = b64png(data.mask);
        resultOverlay.src = b64png(data.overlay);
        sliderBefore.src  = b64png(data.before);
        sliderAfter.src   = b64png(data.after);

        const s   = data.stats || {};
        const aoi = data.aoi   || {};
        statPct.textContent        = (s.change_percentage ?? '—') + (s.change_percentage != null ? '%' : '');
        statConfidence.textContent = (s.confidence ?? 0).toFixed(3);
        statSize.textContent       = s.image_size || '—';
        infoArea.textContent       = aoi.name || '—';
        infoLocation.textContent   = aoi.il ? locationLabel(aoi) : '—';
        infoPreDate.textContent    = aoi.pre_date  || '—';
        infoPostDate.textContent   = aoi.post_date || '—';

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function resetAll() {
        resultData = null;
        resultsSection.style.display = 'none';
        window.scrollTo({ top: 0, behavior: 'smooth' });
        clearQueryMarker();
    }

    // ── View toggle (side-by-side / slider) ────────────────────────────────────

    function setView(view) {
        if (view === 'side') {
            sideView.style.display   = 'grid';
            sliderView.style.display = 'none';
            toggleSide.classList.add('active');
            toggleSlider.classList.remove('active');
        } else {
            sideView.style.display   = 'none';
            sliderView.style.display = 'block';
            toggleSide.classList.remove('active');
            toggleSlider.classList.add('active');
            updateSliderPosition(50);
        }
    }

    let isDragging = false;

    function updateSliderPosition(pct) {
        pct = Math.max(0, Math.min(100, pct));
        sliderOverlayEl.style.width = pct + '%';
        sliderHandle.style.left = pct + '%';
        sliderBefore.style.width = sliderContainer.offsetWidth + 'px';
    }
    function getSliderPct(e) {
        const rect = sliderContainer.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        return ((clientX - rect.left) / rect.width) * 100;
    }
    function onSliderStart(e) { e.preventDefault(); isDragging = true; updateSliderPosition(getSliderPct(e)); }
    function onSliderMove(e)  { if (!isDragging) return; e.preventDefault(); updateSliderPosition(getSliderPct(e)); }
    function onSliderEnd()    { isDragging = false; }

    // ── Downloads ──────────────────────────────────────────────────────────────

    function downloadComposite() {
        if (!resultData) return;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const images = [resultBefore, resultAfter, resultMask, resultOverlay];
        const labels = ['Before', 'After', 'Mask', 'Overlay'];
        const imgW = 256, imgH = 256, pad = 20, labelH = 30;
        const totalW = (imgW * images.length) + (pad * (images.length + 1));
        const totalH = imgH + labelH + (pad * 2);
        canvas.width = totalW; canvas.height = totalH;
        ctx.fillStyle = '#0a0e1a';
        ctx.fillRect(0, 0, totalW, totalH);
        images.forEach((img, i) => {
            const x = pad + (imgW + pad) * i;
            const y = labelH + pad;
            ctx.fillStyle = '#8892b0';
            ctx.font = '600 13px Inter, sans-serif';
            ctx.fillText(labels[i], x, labelH);
            ctx.drawImage(img, x, y, imgW, imgH);
        });
        const link = document.createElement('a');
        link.download = 'change_detection_result.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    }

    // ── Error toast ────────────────────────────────────────────────────────────

    function showError(msg) {
        toastMsg.textContent = msg;
        toast.style.display = 'flex';
        clearTimeout(showError._t);
        showError._t = setTimeout(() => { toast.style.display = 'none'; }, 6000);
    }

    // ── Misc ───────────────────────────────────────────────────────────────────

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        }[c]));
    }

    // ── Wiring ─────────────────────────────────────────────────────────────────

    tabBrowse.addEventListener('click', () => setTab('browse'));
    tabCoords.addEventListener('click', () => setTab('coords'));

    selectIl     .addEventListener('change', onIlChange);
    selectIlce   .addEventListener('change', onIlceChange);
    selectMahalle.addEventListener('change', onMahalleChange);

    btnFind   .addEventListener('click',   onFindAoi);
    inputLat  .addEventListener('keydown', (e) => { if (e.key === 'Enter') onFindAoi(); });
    inputLon  .addEventListener('keydown', (e) => { if (e.key === 'Enter') onFindAoi(); });

    btnDetect  .addEventListener('click', runDetection);
    btnNew     .addEventListener('click', resetAll);
    btnDownload.addEventListener('click', downloadComposite);

    toggleSide  .addEventListener('click', () => setView('side'));
    toggleSlider.addEventListener('click', () => setView('slider'));

    sliderContainer.addEventListener('mousedown',  onSliderStart);
    sliderContainer.addEventListener('touchstart', onSliderStart, { passive: false });
    window.addEventListener('mousemove', onSliderMove);
    window.addEventListener('touchmove', onSliderMove, { passive: false });
    window.addEventListener('mouseup',  onSliderEnd);
    window.addEventListener('touchend', onSliderEnd);

    // Initial state: hide the mahalle field until needed (keeps the cascade tidy).
    fieldMahalle.style.visibility = 'hidden';

    // Boot: load catalog, draw map layers, populate İl dropdown.
    bootstrap();
    setTimeout(() => pickMap.invalidateSize(), 200);
})();

/* ═══════════════════════════════════════════════════════════════════════════
   Upload-to-Dataset modal — self-contained IIFE, does not touch the main
   picker state. On success, full page reload so the new AOI shows up in the
   map + cascade without re-running the catalog/render plumbing.
   ═══════════════════════════════════════════════════════════════════════════ */
(() => {
    'use strict';

    const $ = (s) => document.querySelector(s);

    const modal       = $('#upload-modal');
    const openBtn     = $('#btn-open-upload');
    const closeBtn    = $('#btn-close-upload');
    const cancelBtn   = $('#btn-cancel-upload');
    const form        = $('#upload-form');
    const submitBtn   = $('#btn-submit-upload');
    const submitText  = submitBtn?.querySelector('.btn-text');
    const submitLoad  = $('#upload-loader');
    const errBox      = $('#upload-error');

    if (!modal || !openBtn || !form) return;

    function open() {
        // Always start clean. Defensive against browsers that restore form
        // values across reloads (bfcache / autofill) — otherwise the fields
        // would still hold the previous submission's values when the modal
        // is reopened after a successful upload.
        form.reset();
        errBox.hidden = true;
        errBox.textContent = '';
        modal.hidden = false;
        // Focus the first text input for keyboard users.
        setTimeout(() => form.querySelector('input[name="name"]')?.focus(), 30);
    }

    function close() {
        modal.hidden = true;
    }

    function setLoading(loading) {
        submitBtn.disabled = loading;
        if (submitText) submitText.style.display = loading ? 'none' : '';
        if (submitLoad) submitLoad.style.display = loading ? 'inline-flex' : 'none';
    }

    function showErr(msg) {
        errBox.textContent = msg;
        errBox.hidden = false;
        errBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    openBtn.addEventListener('click', open);
    closeBtn.addEventListener('click', close);
    cancelBtn.addEventListener('click', close);

    // Click outside the panel to dismiss.
    modal.addEventListener('click', (e) => {
        if (e.target === modal) close();
    });

    // Esc to dismiss.
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.hidden) close();
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        errBox.hidden = true;

        const fd = new FormData(form);

        // Client-side sanity check on dates — backend re-validates, but a
        // friendlier message here saves a roundtrip.
        const preDate  = fd.get('pre_date');
        const postDate = fd.get('post_date');
        if (preDate && postDate && preDate > postDate) {
            showErr('Pre date must be on or before post date.');
            return;
        }

        setLoading(true);
        try {
            const res = await fetch('/api/aois', { method: 'POST', body: fd });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(data.detail || data.message || `Upload failed (${res.status})`);
            }
            // Easiest way to surface the new AOI everywhere (map pin, cascade,
            // count badge) without rewiring renderers.
            window.location.reload();
        } catch (err) {
            showErr(err.message || 'Upload failed.');
            setLoading(false);
        }
    });
})();
