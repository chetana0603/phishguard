# PhishGuard Phase 1 Data Audit

## Dataset overview

- Prepared rows: **235,150**
- Phishing rows: **100,300**
- Legitimate rows: **134,850**
- Missing model inputs: **0**
- Invalid URL rows retained: **0**
- Unique registered-domain split groups: **175,509**
- Conflicting-label rows removed: **0**
- Exact duplicate model inputs remaining: **0**
- Normalised duplicate keys remaining: **0**

## Target convention

- `1` = phishing
- `0` = legitimate

## Class summary

| target | class | rows | percentage | unique_domains |
| --- | --- | --- | --- | --- |
| 0 | legitimate | 134850 | 57.35 | 132117 |
| 1 | phishing | 100300 | 42.65 | 43512 |

## URL-shape summary

| statistic | url_length | digit_count | symbol_count |
| --- | --- | --- | --- |
| count | 235150.0 | 235150.0 | 235150.0 |
| mean | 35.29 | 1.87 | 6.27 |
| std | 41.21 | 11.9 | 4.89 |
| min | 14.0 | 0.0 | 4.0 |
| 50% | 28.0 | 0.0 | 5.0 |
| 90% | 50.0 | 5.0 | 8.0 |
| 95% | 73.0 | 10.0 | 10.0 |
| 99% | 144.0 | 27.0 | 18.0 |
| max | 6097.0 | 2011.0 | 775.0 |

## Most frequent domains by class

| class | registered_domain | rows |
| --- | --- | --- |
| legitimate | af.mil | 124 |
| legitimate | ox.ac.uk | 78 |
| legitimate | senate.gov | 71 |
| legitimate | ca.gov | 67 |
| legitimate | nsw.gov.au | 66 |
| legitimate | cam.ac.uk | 53 |
| legitimate | army.mil | 52 |
| legitimate | uscourts.gov | 52 |
| legitimate | marines.mil | 41 |
| legitimate | noaa.gov | 38 |
| phishing | web.app | 5705 |
| phishing | firebaseapp.com | 5544 |
| phishing | repl.co | 3746 |
| phishing | weeblysite.com | 3079 |
| phishing | ipfs.io | 1513 |
| phishing | workers.dev | 1421 |
| phishing | square.site | 1187 |
| phishing | dweb.link | 940 |
| phishing | xsph.ru | 928 |
| phishing | pantheonsite.io | 891 |

## Generated figures

- `reports/figures/class_distribution.png`
- `reports/figures/url_length_distribution.png`
- `reports/figures/domain_frequency.png`

## Safety and scope notes

- No script in Phase 1 opens, renders, or requests any URL.
- The modelling table uses raw URL strings and features derived locally from
  those strings.
- Webpage-source features supplied by the original dataset are intentionally
  excluded from Version 1.
- Raw URLs are not printed in this report, reducing accidental click risk.
