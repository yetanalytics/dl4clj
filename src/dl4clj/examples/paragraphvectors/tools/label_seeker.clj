(ns ^{:doc "see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/paragraphvectors/tools/LabelSeeker.java"}
  dl4clj.examples.paragraphvectors.tools.label-seeker
  (:refer-clojure :exclude [vector])
  (:require [nd4clj.linalg.ops.transforms.transforms :refer (cosine-sim)]
            [dl4clj.models.embeddings.weight-lookup-table :refer (vector)]))

(defn get-scores
  "Accepts a vector representing a document and returns distances to previously trained categories"
  [labels lookup-table v]
  (loop [labels labels
         ret {}]
    (if (empty? labels)
      ret
      (let [label (first labels)
            v-label (vector lookup-table label)]
        (when (nil? v-label)
          (throw (IllegalStateException. (str "Label '" label "' has no known vector"))))
        (recur (rest labels) (assoc ret label (cosine-sim v-label v)))))))
