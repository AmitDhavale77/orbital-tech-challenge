# Orbital Take-Home — Commercial Real-Estate Document Q&A

An AI assistant that helps commercial property lawyers review, cross-reference, and answer questions across bundles of legal documents, with verifiable citations back to the source.

## Language

### Legal / Real-Estate Domain

**Lease**:
The core agreement granting a tenant occupation of a property for a term in return for rent and obligations.
_Avoid_: contract, agreement, tenancy (when referring to the document)

**Landlord**:
The party that grants the lease and holds the reversion.
_Avoid_: lessor, owner

**Tenant**:
The party granted the right to occupy the property under the lease.
_Avoid_: lessee, renter, occupier

**Term**:
The fixed duration the lease grants occupation for.
_Avoid_: length, period, duration

**Rent**:
The periodic sum the tenant pays the landlord for occupation of the property.

**Rent Review**:
A mechanism that resets the rent at defined intervals, with the outcome recorded in a Rent Review Memorandum.
_Avoid_: rent increase, repricing

**Peppercorn**:
A nominal, effectively zero, rent used to keep a lease legally valid without real payment.
_Avoid_: free, zero rent, nominal payment

**Break Clause**:
A right for the landlord or tenant to end the lease early on a defined date, subject to notice and conditions.
_Avoid_: termination clause, exit clause, cancellation

**Title Register**:
The Land Registry record of who owns a property and the charges, restrictions, and rights affecting it.
_Avoid_: title deed, deeds, ownership record

**Title Plan**:
The Land Registry map showing the extent of the property, usually edged in red.
_Avoid_: map, site plan, boundary map

**Deed of Variation**:
A deed that amends the terms of an existing lease, such as the rent or the rights granted.
_Avoid_: amendment, addendum, side letter

**Right**:
A benefit enjoyed over land — such as access or services — granted to the property or to the tenant.
_Avoid_: easement (reserve for the specific land-law sense), permission

**Assignment**:
Transfer of the tenant's entire interest in the lease to a third party.
_Avoid_: transfer, sale, handover

**Subletting**:
The tenant granting a lease of the property, or part of it, to another while keeping its own interest.
_Avoid_: subleasing, underletting, renting out

**Permitted Use**:
The uses of the property that the lease allows.
_Avoid_: usage, purpose

**Charge**:
A security interest, such as a mortgage, registered against the title.
_Avoid_: mortgage, lien, encumbrance

**Restriction**:
A title entry that limits how the property may be dealt with.

**Schedule of Condition**:
A record of the property's condition at the start of the lease that limits the tenant's repair liability.
_Avoid_: survey, condition report

**Document Bundle**:
The full set of documents for a single property, reviewed together by the lawyer.
_Avoid_: documents, files, pack, collection

**Report on Title**:
The synthesised report a lawyer produces summarising findings on a property's title and lease for a transaction.
_Avoid_: summary, write-up (use Certificate of Title for that specific deliverable)

### Application Domain

**Conversation**:
One chat session scoped to a single document bundle, holding an ordered series of messages.
_Avoid_: chat, session, thread

**Message**:
One turn in a conversation, authored by the user or the assistant.
_Avoid_: post, entry

**Document**:
A single uploaded PDF within a conversation, with its extracted text and page count.
_Avoid_: file, PDF, attachment

**Citation**:
A precise, verifiable reference from an answer back to a location in a source — the document, page, and clause.
_Avoid_: source, reference, footnote

**Source**:
The underlying passage of document text used as evidence for an answer.
_Avoid_: citation (that is the rendered reference), context

**Grounding**:
The property of an answer being traceable to, and supported by, specific document evidence.
_Avoid_: sourcing, backing, proof

**Retrieval**:
Selecting the document passages most relevant to a query to feed the model, instead of sending whole documents.
_Avoid_: search, lookup, fetch

**Extraction**:
Pulling structured facts or answers from documents, such as the rent or a break date.
_Avoid_: parsing, scraping
