window.addEventListener('DOMContentLoaded', () => {
  // initialise bulma-carousel (loaded via CDN)
  if (window.bulmaCarousel) {
    bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 1,
      infinite: true,
      autoplay: false,
      loop: true,
      navigation: true,
      pagination: true,
    });
  }

  // pause/play videos when they leave / enter the viewport to save resources
  const vids = document.querySelectorAll('video[data-lazy]');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      const v = e.target;
      if (e.isIntersecting) {
        v.play().catch(() => {});
      } else {
        v.pause();
      }
    });
  }, { threshold: 0.25 });
  vids.forEach((v) => io.observe(v));
});
