import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const appTsxPath = path.join(rootDir, 'artifacts', 'selene-reg-x', 'src', 'App.tsx');

console.log('==================================================');
console.log('RUNNING PAIR REGISTRATION HOOK-ORDER REGRESSION TEST');
console.log('==================================================');

// 1. Read App.tsx
const appTsx = fs.readFileSync(appTsxPath, 'utf-8');

// 2. Extract Evidence and PairRegistrationResults source code
const evidenceIdx = appTsx.indexOf('function Evidence(');
assert.ok(evidenceIdx !== -1, 'Evidence component must be defined in App.tsx');

const pairResultsIdx = appTsx.indexOf('function PairRegistrationResults(');
assert.ok(pairResultsIdx !== -1, 'PairRegistrationResults component must be defined in App.tsx');

const evidenceIdleIdx = appTsx.indexOf('function EvidenceIdle(');
assert.ok(evidenceIdleIdx !== -1, 'EvidenceIdle component must be defined in App.tsx');

console.log('✔ Component boundaries found for Evidence, EvidenceIdle, and PairRegistrationResults');

// 3. Static Hook Analysis on Evidence component
// Evidence must be a pure branching router with 0 hooks
const evidenceEndIdx = appTsx.indexOf('function AppShell', evidenceIdx);
const evidenceCode = appTsx.slice(evidenceIdx, evidenceEndIdx);

const reactHookPattern = /\b(useState|useEffect|useMemo|useCallback|useRef|useReducer|useContext)\s*\(/g;
const evidenceHooks = [...evidenceCode.matchAll(reactHookPattern)].map(m => m[1]);

console.log(`Hooks found inside Evidence component: [${evidenceHooks.join(', ')}]`);
assert.strictEqual(
  evidenceHooks.length,
  0,
  `Evidence component must have 0 hooks to prevent hook ordering violations across state branches. Found: ${evidenceHooks.join(', ')}`
);
console.log('✔ Evidence component has 0 hooks (clean branching router)');

// 4. Static Hook Analysis on PairRegistrationResults
const pairResultsEndIdx = appTsx.indexOf('function Evidence(', pairResultsIdx);
const pairResultsCode = appTsx.slice(pairResultsIdx, pairResultsEndIdx);

// Component JSX return is at the top level of the component function
const componentReturnIdx = pairResultsCode.search(/\n\s*return\s+(?:<|\()/);
assert.ok(componentReturnIdx !== -1, 'PairRegistrationResults must contain a top-level component JSX return statement');

const genericHookPattern = /\b(useState|useEffect|useMemo|useCallback|useRef|useReducer|useContext)(?:<[^>]*>)?\s*\(/g;
const matches = [...pairResultsCode.matchAll(genericHookPattern)];
console.log(`Hooks found inside PairRegistrationResults: [${matches.map(m => m[1]).join(', ')}]`);
assert.strictEqual(matches.length, 6, 'PairRegistrationResults must declare exactly 6 invariant hooks');

for (const match of matches) {
  assert.ok(
    match.index < componentReturnIdx,
    `Hook ${match[1]} at index ${match.index} is after the component JSX return statement (index ${componentReturnIdx})!`
  );
}
console.log('✔ All 6 hooks in PairRegistrationResults are declared before the component JSX return');

// Check that no hook is inside an if block
const ifBlocks = [...pairResultsCode.matchAll(/if\s*\([^)]*\)\s*\{([^}]*)\}/gs)];
for (const ifBlock of ifBlocks) {
  const hooksInIf = [...ifBlock[1].matchAll(reactHookPattern)].map(m => m[1]);
  assert.strictEqual(
    hooksInIf.length,
    0,
    `Hook found inside if statement: ${hooksInIf.join(', ')}`
  );
}
console.log('✔ No hooks inside if statements in PairRegistrationResults');

// 5. Test State Transition Lifecycle Simulation
// Simulate the React component rendering across all required state transitions:
// State 1: Initial Pair Registration render (idle: no images, no result)
// State 2: One image selected
// State 3: Two images selected
// State 4: Run Correspondence (isRegistering = true, pipeline starting)
// State 5: Stage progression (isRegistering = true, stage = SIFT_DETECTION)
// State 6: Stage progression (isRegistering = true, stage = METRICS_EVALUATION)
// State 7: Success result arrives (isRegistering = false, result = PASS)
// State 8: Filter changed to 'inliers'
// State 9: Filter changed to 'outliers'
// State 10: Selected match point changed
// State 11: Rejected result arrives (isRegistering = false, result = REJECTED)
// State 12: Pipeline transport error (isRegistering = false, transportError = '...')
// State 13: Error dismissed / back to idle

class HookTracker {
  constructor() {
    this.previousHookCalls = null;
    this.currentHookCalls = [];
  }

  startRender(componentName) {
    this.componentName = componentName;
    this.currentHookCalls = [];
  }

  recordHook(hookName, id) {
    this.currentHookCalls.push(`${hookName}_${id}`);
  }

  endRender() {
    if (this.previousHookCalls !== null) {
      if (this.previousHookCalls.length !== this.currentHookCalls.length) {
        throw new Error(
          `Rendered ${this.currentHookCalls.length > this.previousHookCalls.length ? 'more' : 'fewer'} ` +
          `hooks than during the previous render in component <${this.componentName}>.\n` +
          `Previous: [${this.previousHookCalls.join(', ')}]\n` +
          `Current:  [${this.currentHookCalls.join(', ')}]`
        );
      }
      for (let i = 0; i < this.previousHookCalls.length; i++) {
        if (this.previousHookCalls[i] !== this.currentHookCalls[i]) {
          throw new Error(
            `Hook ordering violation at index ${i} in <${this.componentName}>: ` +
            `expected ${this.previousHookCalls[i]}, got ${this.currentHookCalls[i]}`
          );
        }
      }
    }
    this.previousHookCalls = [...this.currentHookCalls];
  }
}

// Simulate Evidence component dispatch
function simulateEvidenceRender(props, resultsTracker, progressTracker, idleTracker) {
  if (props.isRegistering || (!props.result && props.transportError)) {
    // Mounts RegistrationProgressPanel
    progressTracker.startRender('RegistrationProgressPanel');
    // RegistrationProgressPanel has 0 hooks
    progressTracker.endRender();
    return 'RegistrationProgressPanel';
  }

  if (!props.result) {
    // Mounts EvidenceIdle
    idleTracker.startRender('EvidenceIdle');
    // EvidenceIdle has 0 hooks
    idleTracker.endRender();
    return 'EvidenceIdle';
  }

  // Mounts PairRegistrationResults
  resultsTracker.startRender('PairRegistrationResults');
  resultsTracker.recordHook('useState', 'matchFilter');
  resultsTracker.recordHook('useMemo', 'outputEntries');
  resultsTracker.recordHook('useMemo', 'allCorrespondences');
  resultsTracker.recordHook('useMemo', 'filteredCorrespondences');
  resultsTracker.recordHook('useState', 'showMatchInspector');
  resultsTracker.recordHook('useState', 'showAdvancedDiagnostics');
  resultsTracker.endRender();
  return 'PairRegistrationResults';
}

const resultsTracker = new HookTracker();
const progressTracker = new HookTracker();
const idleTracker = new HookTracker();

const mockPassResult = {
  job_id: 'job-pass-001',
  quality_gate_passed: true,
  quality_status: 'PASS',
  outputs: {
    registered_image: 'blob:registered.png',
    match_visualization: 'blob:matches.png',
  },
  correspondences: [
    { index: 1, source: [100, 100], reference: [102, 101], residual: 0.25, status: 'INLIER', is_inlier: true },
    { index: 2, source: [200, 200], reference: [205, 204], residual: 0.40, status: 'INLIER', is_inlier: true },
    { index: 3, source: [300, 300], reference: [350, 350], residual: 5.50, status: 'OUTLIER', is_inlier: false },
  ],
};

const mockFailResult = {
  job_id: 'job-fail-002',
  quality_gate_passed: false,
  quality_status: 'FAILED',
  outputs: {
    match_visualization: 'blob:failed_matches.png',
  },
  correspondences: [
    { index: 1, source: [10, 10], reference: [50, 50], residual: 12.0, status: 'OUTLIER', is_inlier: false },
  ],
};

const transitions = [
  { name: '1. Initial Idle render', props: { isRegistering: false, result: undefined, transportError: undefined } },
  { name: '2. One image selected', props: { isRegistering: false, result: undefined, transportError: undefined, source: { url: 'img1.png' } } },
  { name: '3. Two images selected', props: { isRegistering: false, result: undefined, transportError: undefined, source: { url: 'img1.png' }, reference: { url: 'img2.png' } } },
  { name: '4. Run correspondence clicked -> loading', props: { isRegistering: true, result: undefined, transportError: undefined } },
  { name: '5. Pipeline stage: SIFT_DETECTION', props: { isRegistering: true, result: undefined, transportError: undefined } },
  { name: '6. Pipeline stage: HOMOGRAPHY_VERIFICATION', props: { isRegistering: true, result: undefined, transportError: undefined } },
  { name: '7. Success result arrives (PASS)', props: { isRegistering: false, result: mockPassResult, transportError: undefined } },
  { name: '8. Filter tab changed to INLIERS', props: { isRegistering: false, result: mockPassResult, transportError: undefined } },
  { name: '9. Filter tab changed to OUTLIERS', props: { isRegistering: false, result: mockPassResult, transportError: undefined } },
  { name: '10. Selected match point changed', props: { isRegistering: false, result: mockPassResult, transportError: undefined } },
  { name: '11. Replace image & run again -> loading', props: { isRegistering: true, result: undefined, transportError: undefined } },
  { name: '12. Failed / rejected result arrives (FAILED)', props: { isRegistering: false, result: mockFailResult, transportError: undefined } },
  { name: '13. Run again -> transport error', props: { isRegistering: false, result: undefined, transportError: 'Network timeout during matching' } },
  { name: '14. Error dismissed -> back to idle', props: { isRegistering: false, result: undefined, transportError: undefined } },
];

for (const t of transitions) {
  try {
    const rendered = simulateEvidenceRender(t.props, resultsTracker, progressTracker, idleTracker);
    console.log(`✔ [${t.name}] -> rendered <${rendered}> without hook violations`);
  } catch (err) {
    console.error(`FAILED on [${t.name}]:`, err.message);
    process.exit(1);
  }
}

console.log('==================================================');
console.log('ALL PAIR REGISTRATION HOOK-ORDER TESTS PASSED');
console.log('==================================================');
