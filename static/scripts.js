document.addEventListener('DOMContentLoaded', () => {
  const buttons = document.getElementsByClassName('btn-next');

  Array.from(buttons).forEach(button => {
    const targetSelector = button.getAttribute('href');
    const target = document.querySelector(targetSelector);

    if (target) {
      button.addEventListener('click', (e) => {
        e.preventDefault();
        smoothScrollTo(target.offsetTop, 1000); // 1000ms = 1s
      });
    }
  });
});

function smoothScrollTo(targetY, duration) {
  const startY = window.scrollY;
  const distance = targetY - startY;
  const startTime = performance.now();

  function scroll(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = progress < 0.5
      ? 2 * progress * progress
      : -1 + (4 - 2 * progress) * progress;
    window.scrollTo(0, startY + distance * ease);

    if (progress < 1) {
      requestAnimationFrame(scroll);
    }
  }

  requestAnimationFrame(scroll);
}
document.addEventListener("DOMContentLoaded", function() {
    const third = document.getElementById('third');
    if (window.location.hash === "#third" && third.hasAttribute("hidden")) {
      third.removeAttribute("hidden");
    }
  });