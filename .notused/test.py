from Downloader import downloadChannelAudio, transcribeDirAudio
import re
# window.clearInterval(scroll); console.clear(); urls = $$('a'); urls.forEach(function(v,i,a){if (v.id=="video-title-link"){console.log('\t||'+v.title+'\t'+v.href+'||\t')}});
if __name__ == "__main__":
    channel = 'sirswag'


    # test = re.compile('(The Daily Gwei Refuel \d+)?=* - Ethereum Updates.mp4')
    sirSwag = re.compile("This month's news without the bulls%&t")

    downloadChannelAudio(channel, sirSwag, 200, 10)
    transcribeDirAudio('medium.en', channel)
