"use client";
import { useEffect, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { Copy, Check } from "lucide-react";

export default function ConnectPage() {
  const [url, setUrl] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const { protocol, hostname } = window.location;
    setUrl(`${protocol}//${hostname}:3000`);
  }, []);

  const copyUrl = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for browsers that block clipboard without user gesture
    }
  };

  return (
    <div className="max-w-lg mx-auto py-8">
      <h1 className="text-2xl font-bold mb-2">Connect from Quest 3</h1>
      <p className="text-gray-400 mb-8">
        Access this app from your Meta Quest 3 headset over your local network.
      </p>

      {url && (
        <div className="bg-gray-800 rounded-xl p-6 mb-6 text-center">
          <p className="text-sm text-gray-400 mb-3">
            Scan with your phone to get the URL, then send it to your headset:
          </p>
          <div className="inline-block bg-white p-4 rounded-lg mb-4">
            <QRCodeSVG value={url} size={180} />
          </div>
          <div className="flex items-center gap-2 mt-2">
            <span className="flex-1 bg-gray-900 rounded-lg px-4 py-3 text-lg font-mono text-blue-400 text-center break-all">
              {url}
            </span>
            <button
              onClick={copyUrl}
              className="flex items-center gap-1 bg-gray-700 hover:bg-gray-600 px-3 py-3 rounded-lg transition shrink-0"
              title="Copy URL"
            >
              {copied ? (
                <Check className="w-5 h-5 text-green-400" />
              ) : (
                <Copy className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      )}

      <div className="bg-gray-800 rounded-xl p-6 mb-4">
        <h2 className="font-semibold mb-3 text-white">Connection Instructions</h2>
        <ol className="space-y-2 text-sm text-gray-300">
          <li className="flex gap-3">
            <span className="text-blue-400 font-bold shrink-0">1.</span>
            Put on your Meta Quest 3 headset
          </li>
          <li className="flex gap-3">
            <span className="text-blue-400 font-bold shrink-0">2.</span>
            Open the <strong className="text-white">Meta Quest Browser</strong>
          </li>
          <li className="flex gap-3">
            <span className="text-blue-400 font-bold shrink-0">3.</span>
            <span>
              Type this URL in the address bar:{" "}
              <code className="text-blue-400 font-mono">
                {url || "http://<LAN_IP>:3000"}
              </code>
            </span>
          </li>
        </ol>

        <div className="mt-4 pt-4 border-t border-gray-700">
          <p className="text-sm text-gray-400 font-medium mb-2">
            Alternative: Meta Quest phone app
          </p>
          <ol className="space-y-1 text-sm text-gray-300">
            <li className="flex gap-3">
              <span className="text-blue-400 font-bold shrink-0">1.</span>
              Open the <strong className="text-white">Meta Quest</strong> app on
              your phone
            </li>
            <li className="flex gap-3">
              <span className="text-blue-400 font-bold shrink-0">2.</span>
              Scan the QR code above with your phone camera
            </li>
            <li className="flex gap-3">
              <span className="text-blue-400 font-bold shrink-0">3.</span>
              Use{" "}
              <strong className="text-white">Share Link to Headset</strong> to
              open it in your Quest browser
            </li>
          </ol>
        </div>
      </div>

      <div className="bg-amber-950/50 border border-amber-800 rounded-xl p-4 text-sm">
        <p className="font-semibold mb-1 text-amber-300">If HTTP doesn&apos;t load:</p>
        <p className="text-amber-400 mb-2">
          The Meta Quest browser may block HTTP on private networks (Private
          Network Access). Try HTTPS:
        </p>
        <ol className="space-y-1 text-amber-400">
          <li>
            1. Stop the frontend server
          </li>
          <li>
            2. In the <code className="font-mono bg-amber-900/50 px-1 rounded">frontend/</code>{" "}
            folder, run:{" "}
            <code className="font-mono bg-amber-900/50 px-1 rounded">
              npm run dev:https
            </code>
          </li>
          <li>
            3. Use{" "}
            <code className="font-mono bg-amber-900/50 px-1 rounded">
              https://
            </code>{" "}
            instead of{" "}
            <code className="font-mono bg-amber-900/50 px-1 rounded">
              http://
            </code>{" "}
            in the URL above
          </li>
        </ol>
      </div>
    </div>
  );
}
