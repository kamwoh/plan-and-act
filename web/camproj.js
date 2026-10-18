// The one camera projection of the page. The plan video and the path overlay both use it.
// A camera is { pos, forward, up, fovy, width, height }: a pinhole camera that looks along `forward`,
// with `up` as the image up direction and `fovy` as the vertical field of view in degrees.
(function (root) {
  "use strict";
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a) => {
    const n = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  };

  // The pixel [x, y] of a world point, with (0, 0) at the top left of the image. Null when the point is
  // behind the camera.
  function project(cam, p) {
    const f = unit(cam.forward);
    const right = unit(cross(f, cam.up));
    const up = cross(right, f);
    const d = [p[0] - cam.pos[0], p[1] - cam.pos[1], p[2] - cam.pos[2]];
    const depth = dot(d, f);
    if (depth <= 1e-6) return null;
    const focal = (cam.height / 2) / Math.tan((cam.fovy * Math.PI) / 360);
    return [cam.width / 2 + (focal * dot(d, right)) / depth, cam.height / 2 - (focal * dot(d, up)) / depth];
  }

  const api = { project };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CamProj = api;
})(typeof window !== "undefined" ? window : this);
