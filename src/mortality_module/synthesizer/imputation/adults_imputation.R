require(mice)
require(lattice)
require(dplyr)
require(arrow)

set.seed(123)

ind <- read_parquet("./adults_non_imputed_middle_fidelity.parquet")

ind[grepl("^(indicator_)", colnames(ind))] <- lapply(ind[grepl("^(indicator_)", colnames(ind))], as.logical)
ind[grepl("^(mlb_)", colnames(ind))] <- lapply(ind[grepl("^(mlb_)", colnames(ind))], as.logical)

ind[grepl("^(category_)", colnames(ind))] <- lapply(ind[grepl("^(category_)", colnames(ind))], as.factor)

ghq = grepl("^(ordinal_person_ghq)", colnames(ind))
ind[ghq] <- lapply(ind[ghq], factor, order=TRUE, levels=seq(min(ind[colnames(ind)[ghq]], na.rm = TRUE),
                                                            max(ind[colnames(ind)[ghq]], na.rm = TRUE)))
# This way we can disregard any potential shifts in the variable; they all also have the same levels

ind["ordinal_person_sf_1"] <- lapply(ind["ordinal_person_sf_1"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))

ind["ordinal_person_sf_2a"] <- lapply(ind["ordinal_person_sf_2a"], factor, order=TRUE, levels=c(1, 2, 3))
ind["ordinal_person_sf_2b"] <- lapply(ind["ordinal_person_sf_2b"], factor, order=TRUE, levels=c(1, 2, 3))
ind["ordinal_person_sf_3a"] <- lapply(ind["ordinal_person_sf_3a"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
ind["ordinal_person_sf_3b"] <- lapply(ind["ordinal_person_sf_3b"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
ind["ordinal_person_sf_4a"] <- lapply(ind["ordinal_person_sf_4a"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
ind["ordinal_person_sf_4b"] <- lapply(ind["ordinal_person_sf_4b"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
ind["ordinal_person_sf_5"] <- lapply(ind["ordinal_person_sf_5"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
ind["ordinal_person_sf_6a"] <- lapply(ind["ordinal_person_sf_6a"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
ind["ordinal_person_sf_6b"] <- lapply(ind["ordinal_person_sf_6b"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
ind["ordinal_person_sf_6c"] <- lapply(ind["ordinal_person_sf_6c"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
ind["ordinal_person_sf_7"] <- lapply(ind["ordinal_person_sf_7"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))

ind["ordinal_person_financial_situation"] <- lapply(ind["ordinal_person_financial_situation"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
ind["ordinal_person_life_satisfaction"] <- lapply(ind["ordinal_person_life_satisfaction"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5, 6, 7))

# NOTE for some methods values must be shifted to start from 0, converted to ordinals, processed, converted to int (!), and shifted back
min_age <- min(ind["ordinal_person_age"], na.rm = TRUE)
max_age <- max(ind["ordinal_person_age"], na.rm = TRUE)

ind["ordinal_person_age"] <- lapply(ind["ordinal_person_age"],
                                    factor,
                                    order=TRUE,
                                    levels=seq(min_age, max_age))

min_year <- min(ind["ordinal_household_year"], na.rm = TRUE)
max_year <- max(ind["ordinal_household_year"], na.rm = TRUE)

ind["ordinal_household_year"] <- lapply(ind["ordinal_household_year"],
                                        factor,
                                        order = TRUE,
                                        levels=seq(min_year, max_year))


ind["total_individuals"] <- lapply(ind["total_individuals"], factor, order=TRUE, levels=seq(min(ind["total_individuals"]), max(ind["total_individuals"])))
ind["total_children"] <- lapply(ind["total_children"], factor, order=TRUE, levels=seq(min(ind["total_children"]), max(ind["total_children"])))

ind["has_partner"] <- lapply(ind["has_partner"], as.logical)

id_names <- grepl("^(id_)", colnames(ind))  # Boolean mask, not actual values
ids <- ind[id_names]
ind <- ind[ , !id_names]

pred <- quickpred(ind, mincor = 0.01) # about 10 minutes
# current version of quickpred does remove complete columns from the list of vars to be predicted
# automatically

#to_predict <- names(which(colSums(is.na(ind)) > 0)) # contain NA values
#do_not_predict <- colnames(ind)[!colnames(ind) %in% to_predict]

#pred <- quickpred(ind, mincor = 0)
#pred[do_not_predict, ] <- 0

options(future.globals.maxSize=10485760000)

start_time <- Sys.time()
imp <- futuremice(ind,
                  parallelseed = 123,
                  n.core = 128,
                  visitSequence = "monotone",
                  m = 1,
                  maxit = 20,
                  method = "pmm",
                  pred = pred)
end_time <- Sys.time()
end_time - start_time

imputed_data <- complete(imp, "long")

# TODO converting from ordinal to integer/double increases the value by one
# TODO education is an ordinal variable
imputed_data <- cbind(ids, imputed_data)

write.csv(imputed_data, "out20.csv")
