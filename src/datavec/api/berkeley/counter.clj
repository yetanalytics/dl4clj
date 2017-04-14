(ns ^{:doc "A map from objects to doubles. Includes convenience methods for getting, setting, and incrementing element counts. Objects not in the counter will return a count of zero. The counter is backed by a HashMap (unless specified otherwise with the MapFactory constructor). see https://deeplearning4j.org/datavecdoc/org/datavec/api/berkeley/Counter.html"}
    dl4clj.datavec.api.berkeley.counter
  (:import [org.datavec.api.berkeley Counter]))


;; not sure if this ns will be necessary

(def testing-counter (Counter. {:foo 1.0
                                :fooz 2.0
                                :baz 2.0
                                :foobaz 3.0}))
(comment
  (.clear testing-counter)
  testing-counter
  (.setAllCounts testing-counter 5.0)
  (.put testing-counter :baz 7.0 true)
  testing-counter
  (.scale testing-counter 2.0)
  testing-counter)
