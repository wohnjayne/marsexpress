# Exploring the data, qualitatively
just some short notes on the data
see [description](https://kelvins.esa.int/mars-express-power-challenge/data/) of data for information
## DMOP input data
" Detailed Mission Operations Plan files."
See [discussion](https://kelvins.esa.int/mars-express-power-challenge/discussion/18/)
```bash
head context--2008-08-22_2010-07-10--dmop.csv
ut_ms,subsystem
1219363211000,"AXXX301A"
1219364909000,"AAAAF20C1"
1219364914000,"AAAAF57A1"
1219364919000,"AAAAF23G1"
1219364924000,"AAAAF60A1"
1219365615000,"AXXX305A"
1219366035000,"AXXX380A"
1219366635000,"ASEQ4200"
1219367381000,"ATTTF301E"
```

The number of distinct events of the A*** type is pretty limited:
```bash
cut -f 2 context--2008-08-22_2010-07-10--dmop.csv  -d ,|grep  '^"A'|sort|uniq|wc
181     181    2113
```

and if we strip away the "occurrence number" (which increases monotonously for each event name), the same is true for the other events:
```bash
cut -f 2 context--2008-08-22_2010-07-10--dmop.csv  -d ,|grep -v  '^"A'| cut -d . -f 1|sort|uniq|wc
     14      14      88
```

so we can represent this data by approx. 200 input features.

**First approach: in the distinct time interval, check which features are present.**

Catch: by doing this, we loose any temporal relations, e.g. if one feature switches something on and another switches the same thing off again

## EVTF input data
"more events are listed in these event files."

Various events, many of the event descriptions can be found [here](https://www.fd-tasc.info/sm1cmd/output_products/download/RO-ESC-IF-5003_H.pdf). E.g. signal acquisition (AOS) and loss of signal (LOS) are logged for Mars Rovers A and B (MRA/MRB) and other stations (such as Madrid, MAD).
```bash
head context--2008-08-22_2010-07-10--evtf.csv 
ut_ms,description
1219364253000,"MLG_LOS_05_/_RTLT_02373"
1219364356000,"NNO_AOS_00_/_RTLT_02373"
1219365058000,"MLG_LOS_02_/_RTLT_02373"
1219365755000,"NNO_AOS_05_/_RTLT_02373"
1219367159000,"NNO_AOS_10_/_RTLT_02374"
1219368640000,"4000_KM_DESCEND"
1219369280000,"MRB_/_RANGE_06000KM_START"
1219369855000,"OCC_MARS_200KM_START_/_RA_181.68_/_DE_-00.08_/_OMP_(296.35,-46.48)_/_SZA_077"
1219369949000,"OCC_MARS_START_/_RA_181.69_/_DE_-00.08_/_OMP_(299.32,-43.44)_/_SZA_076"
```
Again, the number of events is not as big as it seems, because ther are occurence numbers again. But a little more work is done to strip those, as the formats differn. Yuck!

**First approach: ignore**

The description in the PDF above might give hints on which events might be worth considering in a temporal relationship.
**Unclear: why is this part of the input data? Can something like the time of the signal acquisition to a Rover be known in advance?**

## FTL files
"listing of spacecraft pointing events (Flight Dynamics TimeLine)."
```bash
head context--2008-08-22_2010-07-10--ftl.csv 
utb_ms,ute_ms,type,flagcomms
1219363213000,1219365494000,"EARTH","FALSE"
1219365494000,1219369555000,"EARTH","TRUE"
1219369555000,1219369619000,"EARTH","FALSE"
1219369619000,1219370253000,"SLEW","FALSE"
1219370253000,1219373093000,"NADIR","FALSE"
1219373093000,1219374563000,"SLEW","FALSE"
1219374563000,1219376633000,"EARTH","FALSE"
1219376633000,1219381144000,"EARTH","TRUE"
1219381144000,1219386604000,"EARTH","FALSE"
```

If you strip out the timestamps (from-to of the event), there is not much left:
```bash
cat context--2008-08-22_2010-07-10--ftl.csv |cut -d , -f 3-4|sort|uniq
"ACROSS_TRACK","FALSE"
"D1PVMC","FALSE"
"D1PVMC","TRUE"
"D2PLND","FALSE"
"D3POCM","FALSE"
"D4PNPO","FALSE"
"D5PPHB","FALSE"
"EARTH","FALSE"
"EARTH","TRUE"
"INERTIAL","FALSE"
"INERTIAL","TRUE"
"MAINTENANCE","FALSE"
"NADIR","FALSE"
"RADIO_SCIENCE","TRUE"
"SLEW","FALSE"
"SLEW","TRUE"
"SPECULAR","FALSE"
type,flagcomms
"WARMUP","FALSE"
"WARMUP","TRUE"
```

Todo: check if TRUE and FALSE periods of events make up for the total time (i.e., can we represent one event by a single bit that is 1/0, or do we have to include a third "other/unknown" state?

**First shot: use as discrete inputs without timestamps**
Then, it might make sense to make bins of the duration (e.g. <1 minutes <10 minutes <1h <1day), and use this somehow.

## SAAF events
"Solar aspect angles are expressed with respect to the Sun-MarsExpress line."

Only real-valued numbers in here. Have to check quantitavly.

**First approach: use as real-valued input**

## LTDATA files: 
"long term data including sun-mars distance and solar constant on Mars."

```bash
head context--2008-08-22_2010-07-10--ltdata.csv 
ut_ms,sunmars_km,earthmars_km,sunmarsearthangle_deg,solarconstantmars,eclipseduration_min,occultationduration_min
1219363200000,241938908.363002,355756044.048925,19.5650758165222,522.263999026823,4.16666666666667,27.4
1219449600000,241800159.82512,356303701.299971,19.3900749648847,522.863536732556,1.78333333333333,26.9333333333333
1219536000000,241660298.976336,356843151.495015,19.2147343649455,523.468926197049,0,26.5833333333333
1219622400000,241519333.695108,357374355.342686,19.0390523929577,524.080160732136,0,26.1
1219708800000,241377271.952611,357897265.710581,18.8630292247608,524.697233328856,0,25.75
1219795200000,241234121.813101,358411830.797199,18.6866671679489,525.320136649074,0,25.25
1219881600000,241089891.434286,358917997.988541,18.5099708365253,525.948863016975,0,24.8833333333333
1219968000000,240944589.067687,359415718.063857,18.3329471203306,526.583404410461,0,24.4
1220054400000,240798223.059001,359904949.220077,18.1556049255903,527.223752452419,0,24.0333333333333
```

temporal resolution is very small (compared to other data). smooth curves.
(see quantitative part)

**First approach: use as real-valued input**
