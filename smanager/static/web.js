document.addEventListener("click", (event) => {
  const button = event.target.closest(".tab-button");
  if (!button) {
    return;
  }

  const tabBar = button.closest(".tab-bar");
  if (!tabBar) {
    return;
  }

  tabBar.querySelectorAll(".tab-button").forEach((other) => {
    other.classList.remove("active");
    other.setAttribute("aria-pressed", "false");
  });
  button.classList.add("active");
  button.setAttribute("aria-pressed", "true");
});
