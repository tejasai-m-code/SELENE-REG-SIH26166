import { Router, type IRouter } from "express";
import healthRouter from "./health";
import registrationRouter from "./registration";

const router: IRouter = Router();

router.use(healthRouter);
router.use(registrationRouter);

export default router;
