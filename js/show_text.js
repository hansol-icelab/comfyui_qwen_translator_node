import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "translator.ShowText",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ShowText") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const textarea = document.createElement("textarea");
            textarea.readOnly = true;
            textarea.style.cssText = [
                "width:100%",
                "min-height:60px",
                "resize:vertical",
                "background:var(--comfy-input-bg,#1a1a1a)",
                "color:var(--input-text,#ddd)",
                "border:1px solid var(--border-color,#555)",
                "padding:6px",
                "font-size:11px",
                "font-family:monospace",
                "box-sizing:border-box",
            ].join(";");

            const widget = this.addDOMWidget("output_text", "customtext", textarea, {
                getValue: () => textarea.value,
                setValue: (v) => { textarea.value = v ?? ""; },
            });
            widget.serialize = false;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text?.[0] ?? "";
            const widget = this.widgets?.find(w => w.name === "output_text");
            if (widget) {
                widget.element.value = text;
                widget.value = text;
            }
        };
    },
});
