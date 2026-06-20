# Most Significant Technical Achievement: The Proof of Business Agent at Tuza

As the sole AI Engineer at Tuza, an early-stage startup, I designed and shipped an agentic system that automates **Proof of Business (PoB)** checks for WorldPay, one of the largest card-payment acquirers in the world. It now runs in production for every merchant who onboards digitally through our platform, and the whole WorldPay boarding team relies on its reports to decide what to do next with each merchant. We are rolling it out to a second acquirer, Tide, in the coming weeks.

## The problem and its context

Tuza provides a price-comparison and digital onboarding platform for payment acquirers like WorldPay. A merchant lands on the acquirer's site, compares card-terminal products and rates, and fills in an onboarding form. Before a contract is issued, the acquirer must verify that the merchant is genuine — a step called Proof of Business.

Proof of Business checks that the merchant's application data matches the evidence available publicly on the web, across three things:

- **Trading name** — does the application's trading name match the website or social/product listings?
- **Trading address** — does the application's address match what is published?
- **Nature of business** — is the business actually doing what it claims to do?

Until now, every member of the boarding team did this by hand: searching the web for each merchant and writing up their findings in a Word document. It was slow, hard to keep consistent between assessors, and didn't scale with merchant volume. Our goal was to put a PoB Agent inside Tuza's enterprise platform that assesses each merchant automatically and produces a report in the UI. An assessor can then agree or disagree with the agent on each of the three checks — and those agree/disagree signals feed our evaluations.

## Complexity and constraints

The hardest part was not the code — it was acquiring the domain knowledge. You cannot reliably make an agent orchestrate a process you do not understand, and there was no clean policy document to work from. The real rules lived in the heads of the boarding team, inside a large and bureaucratic bank, so the first challenge was simply finding the right people to talk to.

I ran a series of meetings with the head of the boarding team and an assessor who performs the process daily. To get the most out of short 30-minute sessions, I kept a shared running document of "the rules we think we have understood." Each session we reviewed it, corrected my misunderstandings, and surfaced new or edge-case rules. Over several rounds this turned tacit, scattered knowledge into an explicit, verified process I could build against. Alongside this, we assembled a **golden dataset of 30 merchants with outcome-based ground-truth labels supplied directly by WorldPay**.

The build itself was constrained by the team. Tuza had only four software engineers, which made PR review a serious bottleneck — a problem for agent work, where you need to iterate quickly until evals give you enough confidence to trust the system.

## My approach

My guiding principle was to build the full end-to-end system **and** the eval harness first, then let results and insights drive the iteration.

To get around the review bottleneck and iterate fast, I built the agentic workflow in **n8n** (a no-code agent-building tool) running as a separate microservice. The monolith triggers it and receives the result back as a callback. This was a deliberate trade-off: building the agent natively in our codebase would have been cleaner long-term, but every change would have been stuck behind PR review and slowed discovery to a crawl. n8n let me change prompts and workflow logic in minutes during the phase where speed of learning mattered most.

Once the eval harness was ready, we rolled the agent out to WorldPay as a beta, where they could see the generated PoB PDF. The crucial next step was live feedback. For an agentic system like this, feedback comes in two forms:

1. **Is the assessment correct?** — did the agent reach the right outcome on each check?
2. **Is the rationale right?** — does the language, structure, and tone of its reasoning match what the policy dictates?

I ran weekly iterations. Each week I acted on the feedback to improve the workflow and prompts and bake in more of the process's nuance, then re-evaluated against the eval dataset — which itself grew every week as assessors labelled more cases.

Finally, once performance was good enough, I hardened the system: I rebuilt the whole process as a dedicated package inside our monolith repo that returns a clean JSON response, which we use to generate the PDF. This gave us the long-term maintainability that the n8n prototype had traded away — speed first, production-grade second.

## Impact

The PoB Agent is now in full production. **WorldPay uses it for every merchant who onboards digitally**, and the entire onboarding team uses the agent-generated PDF to decide next actions, replacing the manual web-search-and-Word-document workflow.

We are also going live with a second acquirer, **Tide**, in the next few weeks!

## Reflection

The main lesson I took from this project is that, for production AI systems, the model is not always the hardest part. The real work is turning an ambiguous human process into something explicit enough to evaluate, automate, and trust.

At the start, Proof of Business looked like a classic web-search and reasoning problem. In practice, it became a policy-reconstruction problem. The boarding team had years of judgment encoded in habits, exceptions, and informal thresholds that were not written down anywhere. For example, different merchant types needed to be assessed against different sources: an e-commerce business might be validated through its website, product listings, checkout flow, or delivery information, while a physical business might rely more heavily on address evidence, local listings, and visible trading presence. Even then, the confidence threshold was not the same across trading name, trading address, and nature of business. Thus, the success of the agentic system depended on extracting these rules carefully, validating them with the people who applied them every day, and building the system around that verified policy rather than around what an LLM could plausibly infer.

If I did it again, I would formalise that policy layer earlier. The shared rules document became one of the most important assets in the project, but initially it grew as meeting notes rather than as a structured specification. Next time, I would treat it as a first-class engineering artefact from the start: versioned, testable, linked directly to real merchant examples, and used as the source of truth for prompts, workflow logic, and evaluations. That would have made the agent easier to improve, easier to explain to stakeholders, and safer to extend to new acquirers.

Finally, the trade-off I would still make is prototyping outside the monolith. Using n8n was not the cleanest long-term architecture, but it was the necessary decision for the discovery phase. It let me move at the speed the problem required, learn from real assessor feedback, and only commit to a production implementation once the shape of the system was clear.