import os
import csv

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

RESULTS_ROOT = "Results_ft"
OUTPUT_CSV = os.path.join(RESULTS_ROOT, "metadata.csv")

# Transcriptions dictionary
TRANSCRIPTIONS = {
    "LJ001-0003": "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process",
    "LJ001-0005": "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing",
    "LJ001-0006": "And it is worth mention in passing that, as an example of fine typography,",
    "LJ001-0007": "the earliest book printed with movable types, the Gutenberg, or forty-two line Bible of about fourteen fifty-five,",
    "LJ001-0010": "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress",
    "LJ001-0013": "than in the same operations with ugly ones",
    "LJ001-0019": "and which developed more completely and satisfactorily on the side of the lower-case than the capital letters,",
    "LJ001-0020": "the lower-case being in fact invented in the early Middle Ages",
    "LJ001-0021": "The earliest book printed with movable type, the aforesaid Gutenberg Bible, is printed in letters which are an exact imitation",
    "LJ001-0024": "But the first Bible actually dated",
    "LJ001-0025": "imitates a much freer hand, simpler, rounder, and less spiky, and therefore far pleasanter and easier to read",
    "LJ001-0027": "especially as regards the lower-case letters, and type very similar was used during the next fifteen or twenty years not only by Schoeffer,",
    "LJ001-0031": "In fourteen sixty five Sweynheim and Pannartz began printing in the monastery of Subiaco near Rome,",
    "LJ001-0032": "and used an exceedingly beautiful type, which is indeed to look at a transition between Gothic and Roman",
    "LJ001-0033": "but which must certainly have come from the study of the twelfth or even the eleventh century manuscripts",
    "LJ001-0034": "They printed very few books in this type, three only, but in their very first books in Rome, beginning with the year fourteen sixty eight,",
    "LJ001-0036": "But about the same year Mentelin at Strasburg began to print in a type which is distinctly Roman",
    "LJ001-0037": "and the next year Gunther Zeiner at Augsburg followed suit",
    "LJ001-0038": "while in fourteen seventy at Paris Udalric Gering and his associates turned out the first books printed in France, also in Roman character",
    "LJ001-0041": "It must be said that it is in no way like the transition type of Subiaco,",
    "LJ001-0044": "John of Spires and his brother Vindelin, followed by Nicholas Jenson, began to print in that city,",
    "LJ001-0047": "Of Jenson it must be said that he carried the development of Roman type as far as it can go",
    "LJ001-0050": "and though the famous family of Aldus restored its technical excellence, rejecting battered letters,",
    "LJ001-0051": "and paying great attention to the press work or actual process of printing",
    "LJ001-0055": "some of which, as example, that of Jacobus Rubeus or Jacques le Rouge, is scarcely distinguishable from his",
    "LJ001-0056": "It was these great Venetian printers, together with their brethren of Rome, Milan",
    "LJ001-0058": "and are worthy representatives of the eager enthusiasm for the revived learning of that epoch",
    "LJ001-0059": "the greater part of these Italian printers, it should be mentioned, were Germans or Frenchmen, working under the influence of Italian opinion and aims",
    "LJ001-0062": "Even in Italy most of the theological and law books were printed in Gothic letter,",
    "LJ001-0064": "many of whose types, indeed, like that of the Subiaco works, are of a transitional character",
    "LJ001-0066": "In fact Gunther Zeiner's first type is remarkably like the type of the before-mentioned Subiaco books",
    "LJ001-0069": "This type was introduced into England by Wynkyn de Worde, Caxton's successor,",
    "LJ001-0074": "the best, mostly French or Low Country, was neat and clear, but without any distinction,",
    "LJ001-0075": "the worst, which perhaps was the English, was a terrible falling-off from the work of the earlier presses,",
    "LJ001-0077": "In England about this time, an attempt was made",
    "LJ001-0079": "Caslon's type is clear and neat, and fairly well designed,",
    "LJ001-0080": "he seems to have taken the letter of the Elzevirs of the seventeenth century for his model",
    "LJ001-0081": "The type cast from his matrices is still in everyday use",
    "LJ001-0082": "In spite, however, of his praiseworthy efforts, printing had still one last degradation to undergo",
    "LJ001-0083": "The seventeenth century founts were bad rather negatively than positively"
}

# ------------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------------

rows = []

for folder in os.listdir(RESULTS_ROOT):
    folder_path = os.path.join(RESULTS_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    if folder == "metadata.csv":
        continue

    print(f"Processing folder: {folder}")

    for file in os.listdir(folder_path):
        if not file.endswith(".wav"):
            continue

        utt_id = file.replace(".wav", "")

        if utt_id not in TRANSCRIPTIONS:
            print(f"Skipping {utt_id} (no transcription found)")
            continue

        full_path = os.path.join(folder_path, file)

        rows.append([
            utt_id,
            folder,  # model name = folder name
            full_path,
            TRANSCRIPTIONS[utt_id]
        ])

# Write CSV
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["utt_id", "model", "filepath", "text"])
    writer.writerows(rows)

print(f"\nMetadata saved to {OUTPUT_CSV}")
print(f"Total entries: {len(rows)}")
