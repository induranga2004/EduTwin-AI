export interface BreakoutRoomInvitation {
  sequence: number;
}

export interface UserIsInvitedSubscriptionResponse {
  breakoutRoom: BreakoutRoomInvitation[];
}
