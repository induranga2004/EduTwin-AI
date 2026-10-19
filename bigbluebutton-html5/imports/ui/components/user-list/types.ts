import { Meeting } from '/imports/ui/Types/meeting';

export interface UserListComponentProps {}

export interface UserAggregateCountSubscriptionResponse {
  user_aggregate: {
    aggregate: {
      count: number;
    };
  };
}

export interface MeetingPermissionsSubscriptionResponse {
  meeting: Meeting[];
}
