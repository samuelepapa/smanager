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

function setColumnVisibility(columnKey, visible) {
  document.querySelectorAll(`[data-column="${columnKey}"]`).forEach((column) => {
    column.classList.toggle("hidden", !visible);
  });
}

function setColumnsMenuOpen(control, open) {
  const button = control.querySelector("#columns-button");
  const menu = control.querySelector(".columns-menu");
  if (!button || !menu) {
    return;
  }

  menu.classList.toggle("hidden", !open);
  button.setAttribute("aria-expanded", open ? "true" : "false");
}

async function saveDashboardColumns(control) {
  const saveStatus = control.querySelector("[data-column-save-status]");
  const columns = Array.from(control.querySelectorAll(".column-toggle:checked"))
    .map((toggle) => toggle.dataset.column);

  if (saveStatus) {
    saveStatus.textContent = "Saving…";
  }

  try {
    const response = await fetch(control.dataset.saveUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dashboard_columns: columns }),
    });
    if (!response.ok) {
      throw new Error("save failed");
    }

    const result = await response.json();
    const savedColumns = new Set(result.dashboard_columns || columns);
    control.querySelectorAll(".column-toggle").forEach((toggle) => {
      const visible = savedColumns.has(toggle.dataset.column);
      toggle.checked = visible;
      setColumnVisibility(toggle.dataset.column, visible);
    });
    if (saveStatus) {
      saveStatus.textContent = "Saved";
    }
  } catch (_error) {
    if (saveStatus) {
      saveStatus.textContent = "Could not save columns";
    }
  }
}

document.addEventListener("click", (event) => {
  const columnsButton = event.target.closest("#columns-button");
  if (columnsButton) {
    const control = columnsButton.closest("[data-columns-control]");
    if (control) {
      const menu = control.querySelector(".columns-menu");
      setColumnsMenuOpen(control, menu && menu.classList.contains("hidden"));
    }
    return;
  }

  if (!event.target.closest("[data-columns-control]")) {
    document.querySelectorAll("[data-columns-control]").forEach((control) => {
      setColumnsMenuOpen(control, false);
    });
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }

  document.querySelectorAll("[data-columns-control]").forEach((control) => {
    setColumnsMenuOpen(control, false);
  });
});

document.addEventListener("change", (event) => {
  if (!event.target.matches(".column-toggle")) {
    return;
  }

  const control = event.target.closest("[data-columns-control]");
  if (!control) {
    return;
  }

  const toggles = Array.from(control.querySelectorAll(".column-toggle"));
  if (!toggles.some((toggle) => toggle.checked)) {
    event.target.checked = true;
  }

  toggles.forEach((toggle) => {
    setColumnVisibility(toggle.dataset.column, toggle.checked);
  });
  saveDashboardColumns(control);
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
