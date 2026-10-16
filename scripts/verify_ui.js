/**
 * Drive the running app in a real browser and capture what it looks like.
 *
 * Not a unit test: this exists to answer questions a test client cannot - does
 * the page render, do the panels populate, do the images decode, does the wipe
 * respond - and to produce the screenshots used to review the interface. It also
 * fails loudly on any console error, because a React error boundary that
 * silently swaps in a fallback is exactly the kind of breakage a screenshot
 * alone would not reveal.
 *
 * Usage (via scripts/verify_ui.sh, which starts the service first):
 *   STRATA_VERIFY_URL=http://127.0.0.1:8077/ node scripts/verify_ui.js
 */

const path = require("path");
const { chromium } = require("playwright-core");

const URL_BASE = process.env.STRATA_VERIFY_URL || "http://127.0.0.1:8077/";
const CHROME =
  process.env.STRATA_CHROME ||
  "C:/Users/User/AppData/Local/ms-playwright/chromium-1248/chrome-win64/chrome.exe";
const OUT = path.resolve(__dirname, "..", "artifacts", "screens");

const problems = [];
const shots = [];

async function shot(page, name, opts = {}) {
  const file = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: file, ...opts });
  shots.push(name);
  console.log(`    captured ${name}`);
}

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, args: ["--no-proxy-server"] });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 2,
  });

  page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const text = m.text();
    // A missing favicon is noise, not a defect; anything else is real.
    if (/favicon/i.test(text)) return;
    problems.push(`console error: ${text}`);
  });
  page.on("requestfailed", (r) => {
    if (/favicon/i.test(r.url())) return;
    problems.push(`request failed: ${r.url()} (${r.failure()?.errorText})`);
  });

  console.log("  opening the site picker");
  await page.goto(URL_BASE, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(2000);
  await shot(page, "01-sites");

  const cards = await page.locator("main button").count();
  if (cards < 3) problems.push(`only ${cards} buttons on the picker; expected the site cards`);

  console.log("  opening a site");
  await page.locator('button:has-text("Loading Bay 4")').first().click();
  await page.waitForTimeout(2500);
  await shot(page, "02-baseline");

  console.log("  running a rescan (this is a real scan)");
  await page.locator('button:has-text("Run rescan")').first().click();

  await page.waitForFunction(
    () => {
      const t = document.body.innerText;
      return (
        /changes|No changes detected|Could not compare/.test(t) &&
        !/Resolving events/.test(t)
      );
    },
    // Playwright's signature is (fn, arg, options): the timeout has to be the
    // third argument, or the default 30 s silently applies and a real scan -
    // which takes longer than that on a cold model load - looks like a failure.
    null,
    { timeout: 240000 },
  );
  await page.waitForTimeout(4500);

  await shot(page, "03-result");
  await shot(page, "04-result-full", { fullPage: true });

  console.log("  switching to the aligned compare view");
  const compare = page.locator('button:has-text("Compare")').first();
  if (await compare.count()) {
    await compare.click();
    await page.waitForTimeout(1500);
    await shot(page, "05-compare");
  } else {
    problems.push("the Compare view was not offered");
  }

  console.log("  selecting a change to locate it on the frame");
  const locate = page.locator('button[aria-label^="Locate"]').first();
  if (await locate.count()) {
    await locate.click();
    await page.waitForTimeout(1200);
    await shot(page, "06-selected");
  } else {
    problems.push("no change could be located on the frame");
  }

  console.log("  expanding the evidence and the reasoning");
  const evidence = page.locator('button:has-text("evidence")').first();
  if (await evidence.count()) {
    await evidence.click();
    await page.waitForTimeout(600);
  }
  const expand = page.locator('button:has-text("expand all")').first();
  if (await expand.count()) {
    await expand.click();
    await page.waitForTimeout(800);
  }
  await shot(page, "07-evidence", { fullPage: true });

  console.log("  recording the change set");
  const record = page.locator('button:has-text("Record change set")').first();
  if (await record.count()) {
    await page.locator("#gate-note").fill("Checked against the pick list.");
    await record.click();
    await page.waitForTimeout(2500);
    await shot(page, "08-recorded");
  } else {
    problems.push("the approval gate did not offer to record the change set");
  }

  console.log("  narrowing the viewport to check the narrow layout");
  await page.setViewportSize({ width: 430, height: 900 });
  await page.waitForTimeout(1200);
  await shot(page, "09-narrow");
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.waitForTimeout(600);

  await browser.close();

  console.log(`\n  ${shots.length} screenshots in artifacts/screens/`);
  if (problems.length) {
    console.log(`\n  ${problems.length} PROBLEM(S):`);
    for (const p of problems) console.log(`    - ${p}`);
    process.exit(1);
  }
  console.log("  no console errors, no failed requests");
})().catch((e) => {
  console.error(`\n  FAILED: ${e.message}`);
  if (problems.length) for (const p of problems) console.error(`    - ${p}`);
  process.exit(1);
});
