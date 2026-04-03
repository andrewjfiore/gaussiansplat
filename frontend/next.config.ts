import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // Allow large video uploads through the rewrite proxy (default is 10 MB)
    middlewareClientMaxBodySize: "500mb",
  },
  webpack: (config) => {
    // PlayCanvas's "module" field points to unbundled source (83MB barrel export).
    // Force Webpack to use the pre-built UMD/CJS bundle instead.
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...config.resolve.alias,
      playcanvas: "playcanvas/build/playcanvas.mjs",
    };
    return config;
  },
  async rewrites() {
    // NOTE: These rewrites run server-side — they work fine when Quest loads
    // pages from http://192.168.1.x:3000 because the Next.js server itself
    // proxies to localhost:8000. WebSocket connections are handled separately:
    // useWebSocket.ts builds ws://<window.location.hostname>:8000/... so they
    // also use the correct host whether accessing locally or over LAN.
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/ws/:path*",
        destination: "http://localhost:8000/ws/:path*",
      },
    ];
  },
};

export default nextConfig;
