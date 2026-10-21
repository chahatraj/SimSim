# Negotiation

Two agents negotiate the price of a used bike. The seller's persona is the
manipulated condition; we measure the buyer's trust before and after (survey),
the agreed price (probe), and cooperation (judge). The buyer's reasoning is
private.

```bash
simsim run examples/configs/negotiation.yaml --dry-run
simsim run examples/configs/negotiation.yaml --workers 8 --view
simsim summarize runs/negotiation --by style --metric judge.cooperation
```

The key parts of the config:

```yaml
grid:
  style: {neutral: "", tough: "You are a hard bargainer who rarely gives ground."}
agents:
  - name: Buyer
    output_schema:
      thoughts: {type: str, description: "private; the seller never sees this"}
      reply: str
  - name: Seller
    persona: "You are selling your used bike. ${style}"
environment:
  type: Conversation
  private: {Buyer: "You will not pay more than $170.", Seller: "You will not sell below $110."}
probes:
  - {type: survey, question: "I trust the other person in this negotiation.", options: likert7, at: [0, end]}
  - {type: probe, question: "What final price, if any, did you agree on?",
     schema: {agreed: bool, price: [int, 0, 1000]}, at: end}
```

A real transcript (Qwen3.5-4B, tough seller) shows why the private field is
useful: the buyer privately reasons *"the seller was offended by my initial offer
but quickly agreed to mine at $170"*, while the seller had in fact not agreed,
which is a misreading you can only see because the reasoning was logged.

Analysis:

```python
from simsim.analysis import load_runs, load_probes
runs = load_runs("runs/negotiation")
prices = load_probes("runs/negotiation").query("name == 'probe'")[["style", "seed", "agent", "answer.price"]]
trust = load_probes("runs/negotiation").query("name == 'survey'")
trust.pivot_table(index="style", columns="t", values="expected")
```
