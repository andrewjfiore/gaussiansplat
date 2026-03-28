import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow large video uploads (default is 1MB which breaks everything)
  serverActions: {
    bodySizeLimit: "500mb",
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
