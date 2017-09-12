(ns dl4clj.core)

(defn foo
  "I don't do a whole lot."
  [x]
  (println x "Hello, World!"))

;; add a basic example here in a single fn to illistrate the flow
;; add in printlns and listeners and everything

;; look into refactoring with list*
;; https://clojuredocs.org/clojure.core/list*

#_(defn eval-model-whole-ds
  ;; move to core, update then
  "evaluate the model performance on an entire data set and print the final result

  :mln (multi layer network), a trained mln you want to get classification stats for

  :eval-obj (evaler), the object created by new-classification-evaler

  :iter (iter), the dataset iterator which has the data you want to evaluate the model on

  :lazy-data (lazy-seq), a lazy sequence of dataset objects

  you should supply either a dl4j dataset-iterator (:iter) or a lazy-seq (:lazy-data), not both

  returns the evaluation object"
  [& {:keys [mln evaler iter lazy-data]
      :as opts}]
  (let [ds-iter (if (contains? opts :lazy-data)
                  (new-lazy-iter lazy-data)
                  (reset-iterator! iter))]
    (while (has-next? ds-iter)
      (let [nxt (next-example! ds-iter)
            prediction (output :mln mln :input (get-features nxt))]
        (eval-classification!
         :evaler evaler
         :labels (get-labels nxt)
         :network-predictions prediction))))
  (println (get-stats :evaler evaler))
  evaler)
