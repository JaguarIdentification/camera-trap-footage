# Publish the native FiftyOne snapshot

The private Hub Distribution will be produced with FiftyOne's
`push_to_hub()` utility and verified by loading it back with
`load_from_hub()`. This preserves the curated dataset schema, labels, media,
and saved views as one loadable artifact instead of maintaining a second
hand-authored export format.
