class Sanitizer:
    @classmethod
    def age(cls, data, age_min, age_max):
        # non negative, less than 100?
        pass

    @classmethod
    def sex(cls, data, values):
        # m and f
        pass

    @classmethod
    def country(cls, data, values):
        # uk only
        pass

    @classmethod
    def weights(cls, data):
        # positive, no zeros
        pass

    @classmethod
    def sex_in_couple(cls):
        # only males and females, no same sex
        pass

    @classmethod
    def household_roles(cls):
        # one head, one partner? + kids
        pass

    @classmethod
    def child_adult(cls):
        # Child/Adult indicator caind should make sense
        pass
