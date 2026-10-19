"""v1.15 page: the Execution entry shows the frames of the run as a strip. A click opens a frame large."""
import subprocess

from tests.test_page_plan import WEB, _body, _web


def test_every_page_script_parses():
    for path in sorted(WEB.glob("*.js")):
        subprocess.run(["node", "--check", str(path)], check=True)


def test_the_execution_entry_has_the_strip_and_keeps_the_sent_frames():
    end = _body(_web("app.js"), "function onExecEnd(ev)")
    assert "item.append(frameStrip(frames))" in end
    assert "runFrames[planRunId] = frames.filter((f) => f.sent)" in end


def test_each_frame_has_a_label_a_mark_and_opens_large():
    js = _web("app.js")
    assert 'return `${f.at === "stop" ? "stop" : "kf"} ${f.keyframe}/${f.of}`;' in _body(js, "function frameLabel(f)")
    strip = _body(js, "function frameStrip(frames)")
    assert '`kf-thumb${f.sent ? " sent" : ""}`' in strip and 'img.loading = "lazy";' in strip  # a folded round loads none
    assert "openFrame(f)" in strip
    assert '$("frame_dialog").showModal();' in _body(js, "function openFrame(f)")
    html, css = _web("index.html"), _web("style.css")
    assert '<dialog id="frame_dialog"' in html and 'id="frame_img"' in html and 'id="frame_close"' in html
    assert ".kf-thumb.sent img { border-color: var(--accent); }" in css


def test_a_live_frame_gets_the_url_of_its_file():
    src = _body(_web("app.js"), "function frameSrc(f)")
    assert "f.url ||" in src and "/images/${encodeURIComponent(f.file" in src


def test_the_step_details_list_the_frames_before_the_end_images():
    js = _web("app.js")
    images = _body(js, "function stepImageList(key)")
    assert images.index("roundFrames[round]") < images.index("end of the last round")
    assert "roundFrames[chatRound] = (ev.parent && runFrames[ev.parent]) || null;" in js
