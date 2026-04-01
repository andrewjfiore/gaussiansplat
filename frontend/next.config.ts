import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // Allow large video uploads through the rewrite proxy (default is 10 MB)
    middlewareClientMaxBodySize: "500mb",
  },
  async rewrites() {
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
