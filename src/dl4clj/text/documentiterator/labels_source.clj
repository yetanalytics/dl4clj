#_(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/text/documentiterator/LabelsSource.html"}
  dl4clj.text.documentiterator.labels-source
  (:import [org.deeplearning4j.text.documentiterator LabelsSource]))

#_(defn labels-source
  "Build LabelsSource, optionally using externally defined list of string labels."
  ([]
   (LabelsSource.))
  ([labels]
   (LabelsSource. labels)))

#_(defn labels-source-from-template
  "Build LabelsSource using string template."
  [^String template]
  (LabelsSource. ^java.util.List template))

#_(defn get-labels
  "This method returns the list of labels used by this generator instance."
  [^LabelsSource this]
  (into [] (.getLabels this)))

#_(defn get-number-of-labels-used
  "This method returns number of labels used up to the method's call"
  [^LabelsSource this]
  (.getNumberOfLabelsUsed this))

#_(defn next-label
  "Returns next label."
  [^LabelsSource this]
  (.nextLabel this))

#_(defn reset
  "This method should be called from Iterator's reset() method, to keep labels in sync with iterator"
  [^LabelsSource this]
  (.reset this))

#_(defn store-label
  "This method is intended for storing labels retrieved from external sources."
  [^LabelsSource this label]
  (.storeLabel this label))

#_(comment

 (labels-source ["label1" "label2"])

)
