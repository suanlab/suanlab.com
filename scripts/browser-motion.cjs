// Measure the readable, settled UI while continuous decorative loops keep running.
async function settleMotion(page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await Promise.allSettled(document.getAnimations().filter(animation => animation.playState === 'running'
      && animation.effect?.getTiming().iterations !== Infinity).map(animation => animation.finished));
  });
}
module.exports = { settleMotion };
