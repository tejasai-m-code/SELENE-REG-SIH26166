import { spawn, execSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const backendDist = path.join(rootDir, "artifacts", "api-server", "dist", "index.mjs");
const frontendDir = path.join(rootDir, "artifacts", "selene-reg-x");
const viteBin = path.join(frontendDir, "node_modules", "vite", "bin", "vite.js");

import { createConnection } from "node:net";

function isPortInUse(port) {
  return new Promise((resolve) => {
    const socket = createConnection({ port, host: "127.0.0.1" });
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.once("error", () => {
      resolve(false);
    });
  });
}

if (!existsSync(backendDist)) {
  console.log("Backend build not found. Building api-server...");
  execSync(`"${process.execPath}" "${path.join(rootDir, "artifacts", "api-server", "build.mjs")}"`, { stdio: "inherit" });
}

console.log("==================================================");
console.log("SELENE-REG-X — Starting Backend & Frontend...");
console.log("==================================================");

const backendInUse = await isPortInUse(5000);
const frontendInUse = await isPortInUse(5173);

if (backendInUse) {
  console.log("[Notice] Port 5000 is already active. Existing backend instance detected at http://localhost:5000");
}
if (frontendInUse) {
  console.log("[Notice] Port 5173 is already active. Existing frontend instance detected at http://localhost:5173");
}

let backend = null;
if (!backendInUse) {
  // 1. Start Backend on Port 5000
  backend = spawn(process.execPath, [backendDist], {
    cwd: rootDir,
    env: {
      ...process.env,
      PORT: "5000",
      PYTHONPATH: path.join(rootDir, "artifacts", "api-server", "python"),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  backend.stdout.on("data", (data) => {
    const line = data.toString().trim();
    if (line) console.log(`[backend] ${line}`);
  });

  backend.stderr.on("data", (data) => {
    const line = data.toString().trim();
    if (line) console.error(`[backend-err] ${line}`);
  });
}

// 2. Start Frontend on Port 5173
let frontend = null;
if (!frontendInUse) {
  frontend = spawn(
    process.execPath,
    [viteBin, "--config", "vite.config.ts", "--host", "0.0.0.0"],
    {
      cwd: frontendDir,
      env: {
        ...process.env,
        PORT: "5173",
        BASE_PATH: "/",
        API_SERVER_URL: "http://localhost:5000",
      },
      stdio: ["ignore", "pipe", "pipe"],
    }
  );

  frontend.stdout.on("data", (data) => {
    const line = data.toString().trim();
    if (line) console.log(`[frontend] ${line}`);
  });

  frontend.stderr.on("data", (data) => {
    const line = data.toString().trim();
    if (line) console.error(`[frontend-err] ${line}`);
  });
}

console.log(`Backend accessible at http://localhost:5000 ${backendInUse ? '(already running)' : '(launched)'}`);
console.log(`Frontend accessible at http://localhost:5173 ${frontendInUse ? '(already running)' : '(launched)'}`);
console.log("==================================================");

if (backend) {
  backend.on("close", (code) => {
    console.log(`[backend] exited with code ${code}`);
    if (code !== 0 && code !== null) {
      if (frontend) frontend.kill();
      process.exit(code);
    }
  });
}

if (frontend) {
  frontend.on("close", (code) => {
    console.log(`[frontend] exited with code ${code}`);
    if (code !== 0 && code !== null) {
      if (backend) backend.kill();
      process.exit(code);
    }
  });
}

function shutdown() {
  console.log("\nShutting down spawned services...");
  if (backend) backend.kill();
  if (frontend) frontend.kill();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// Keep event loop alive
setInterval(() => {}, 1000 * 60 * 60);
