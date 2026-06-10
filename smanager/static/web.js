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

  if (button.dataset.paramTab) {
    const selectedKey = button.dataset.paramTab;
    document.querySelectorAll(".param-column").forEach((column) => {
      const visible = selectedKey === "all" || column.dataset.paramKey === selectedKey;
      column.classList.toggle("hidden", !visible);
    });
  }
});

document.addEventListener("change", (event) => {
  if (!event.target.matches(".job-selector")) {
    return;
  }

  const selectedCount = document.querySelectorAll(".job-selector:checked").length;
  const bulkActions = document.querySelector(".bulk-actions");
  if (!bulkActions) {
    return;
  }

  bulkActions.classList.toggle("hidden", selectedCount === 0);
});

document.addEventListener("submit", (event) => {
  const submitter = event.submitter;
  if (!submitter || !submitter.dataset.confirm) {
    return;
  }

  const form = event.target;
  const dialog = document.getElementById("confirm-dialog");
  const message = document.getElementById("confirm-message");
  if (!dialog || !message) {
    return;
  }

  event.preventDefault();
  message.textContent = submitter.dataset.confirm;
  dialog.showModal();

  const onClose = () => {
    dialog.removeEventListener("close", onClose);
    if (dialog.returnValue !== "confirm") {
      return;
    }

    if (submitter.name) {
      const hiddenSubmit = document.createElement("input");
      hiddenSubmit.type = "hidden";
      hiddenSubmit.name = submitter.name;
      hiddenSubmit.value = submitter.value;
      form.appendChild(hiddenSubmit);
    }
    form.submit();
  };
  dialog.addEventListener("close", onClose);
});

document.addEventListener("click", async (event) => {
  const copyTarget = event.target.closest("[data-copy]");
  if (!copyTarget || copyTarget.matches("[data-confirm]")) {
    return;
  }

  const value = copyTarget.dataset.copy;
  if (!value || value === "-") {
    return;
  }

  try {
    await navigator.clipboard.writeText(value);
    copyTarget.classList.add("copied");
    setTimeout(() => copyTarget.classList.remove("copied"), 900);
  } catch (_error) {
    copyTarget.classList.remove("copied");
  }
});
