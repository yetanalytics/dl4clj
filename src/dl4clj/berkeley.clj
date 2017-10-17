(ns ^{:doc "see https://deeplearning4j.org/doc/org/deeplearning4j/berkeley/Pair.html"}
    dl4clj.berkeley
  (:import [org.deeplearning4j.berkeley Pair]))

(defn new-pair
  "creates a pair out of the two supplied args"
  [& {:keys [p1 p2]}]
  (Pair. p1 p2))

(defn get-first
  "returns the first element of the pair"
  [pair]
  (.getFirst pair))

(defn get-second
  "returns the second element of the pair"
  [pair]
  (.getSecond pair))

(defn reverse-pair
  "reverse the order of the elements within the pair"
  [pair]
  (.reverse pair))

(defn set-first!
  "sets the first element within an existing pair

  returns the mutated pair"
  [& {:keys [new-f pair]}]
  (doto pair (.setFirst new-f)))

(defn set-second!
  "sets the second element within an existing pair

  returns the mutated pair"
  [& {:keys [new-s pair]}]
  (doto pair (.setSecond new-s)))
