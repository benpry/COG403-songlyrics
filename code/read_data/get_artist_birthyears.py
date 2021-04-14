import requests
import bs4
import string
import csv

ARTISTS_PATH = "../../data/processed/artists.csv"
OUTPUT_PATH = "../../data/processed/artist_birthyears.csv"

if __name__ == "__main__":

    # Function to lookup birthyears
    def lookup(name):
        str(name)
        print('Searching ', name)
        # Check if artist has a wikipedia page
        url_end = name.replace(" ", "_")
        url = "https://en.wikipedia.org/wiki/" + url_end
        response = requests.get(url)
        if response.status_code == 200:
            # Collect text from the first 2 paragraphs on their page
            html = bs4.BeautifulSoup(response.text, 'html.parser')
            paragraphs = html.select("p")
            intro = '\n'.join([ para.text for para in paragraphs[0:2]])
            # Search for first occurence of word "born"
            b = intro.find(" born ")
            if b != -1:
                # Collect words from the 150 characters following "born"
                remove = dict.fromkeys(map( ord, string.punctuation))
                txt = intro[b:b+150].translate(remove).lower()
                txt_split = txt.split()[1:]
                print(txt_split)
                # Grab birthyear from those words
                year = ""
                i = 0
                while year == "" and i < len(txt_split):
                    # Set current word
                    word = txt_split[i]
                    i += 1
                    # Look for a year
                    if word.isnumeric() and  4 <= len(word) and len(word) <= 6:
                        # Any 4 digit number is  potential year
                        if len(word) == 4:
                            potential_year = int(word)
                        # If there's a 5 or 6 digit number - it's possibly a year with footnote
                        elif word.isnumeric() and ( len(word) == 5 or len(word) == 6 ):
                            # Check if the first 4 digits form a valid year
                            potential_year = int(word[0:4])
                        # Check that the number is a valid year
                        if potential_year > 1900 and potential_year < 2020:
                            return potential_year
                        

        # Returns 0 if birthyear wasn't found
        return 0


    # Get all artists 
    artists = []
    fp = open(ARTISTS_PATH, "r", encoding="utf-8")
    lines = fp.readlines()
    artists = [line.rstrip('\n') for line in lines]
    
    # Get birthyears
    artist_birthyears = []
    # For running in batches:
    # for artist in artists[14000:]:
    for artist in artists:
        birthyear = lookup(artist)
        if birthyear != 0:
            artist_birthyears.append([artist, birthyear])
    
    # Write to file
    # For running in batches:
    # with open(OUTPUT_PATH, 'a') as outfile:
    with open(OUTPUT_PATH, 'w') as outfile:
        mywriter = csv.writer(outfile)
        for i in artist_birthyears:
            mywriter.writerow(i)