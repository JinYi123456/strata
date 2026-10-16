/** @type {import('next').NextConfig} */
const nextConfig = {
  // A static export is deliberate: the service already serves this bundle from
  // the same origin as the API, so there is one deployable artefact and one URL
  // rather than a front end and a back end that have to be kept in step.
  output: "export",
  // No image optimiser exists in a static export, and every image here is
  // already served by the API at its natural size.
  images: { unoptimized: true },
  // Emit `index.html` per route so a plain static file server resolves them.
  trailingSlash: true,
  reactStrictMode: true,
};

export default nextConfig;
