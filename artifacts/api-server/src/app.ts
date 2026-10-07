import express, { type Express } from "express";
import path from "node:path";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
import { existsSync } from "node:fs";

const outputsDir = existsSync(path.resolve(process.cwd(), "python/worker.py"))
  ? path.resolve(process.cwd(), "data/outputs")
  : path.resolve(process.cwd(), "artifacts/api-server/data/outputs");

const demoDir = existsSync(path.resolve(process.cwd(), "demo_data"))
  ? path.resolve(process.cwd(), "demo_data")
  : path.resolve(process.cwd(), "../../demo_data");

app.use(
  "/api/outputs",
  express.static(outputsDir),
);
app.use(
  "/api/demo_data",
  express.static(demoDir),
);

app.use("/api", router);

export default app;
